"""Native ONNX Runtime QDQ INT8 calibration, scale matching, and explanations.

Native Percentile uses ONNX Runtime's histogram estimator, which differs from
the exact NumPy percentile used in simulation.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import onnxruntime as ort
ort.disable_telemetry_events()
import onnx
from onnx import numpy_helper, helper, TensorProto
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from skl2onnx import to_onnx
from calibration_explanations import FEATURES
from repaired_cohort_study import CATEGORICAL, prepare, grouped_attributions, explain_stats, Quantized

class Rows(CalibrationDataReader):
    def __init__(self, name, rows):
        self.name, self.rows, self.index = name, rows, 0
    def get_next(self):
        if self.index >= len(self.rows):
            return None
        row=self.rows[self.index:self.index+1]
        self.index += 1
        return {self.name:row.astype(np.float32)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--draws',type=int,default=12)
    p.add_argument('--max-iter',type=int,default=90)
    p.add_argument('--linear-only',action='store_true',
                   help='Quantize MatMul/Add/Relu, leaving the probability output ops FP32')
    p.add_argument('--matmul-only',action='store_true',
                   help='Quantize only MatMul inputs and weights; exclude MatMul output QDQ')
    p.add_argument('--output',type=Path,default=Path('repaired_cohort_results'))
    args=p.parse_args()
    if args.linear_only and args.matmul_only:
        p.error('Choose at most one quantization scope')
    out=args.output
    out.mkdir(exist_ok=True)
    raw,y=prepare(Path('repaired_cohort_results/cache'))
    ix=np.arange(len(y))
    tr,rem=train_test_split(ix,train_size=35000,random_state=824,stratify=y)
    te,rem=train_test_split(rem,train_size=10000,random_state=825,stratify=y[rem])
    va,cal=train_test_split(rem,train_size=3000,random_state=826,stratify=y[rem])
    num=raw[FEATURES].astype(float).to_numpy()
    cat=raw[CATEGORICAL].fillna('Missing').astype(str)
    sc=StandardScaler().fit(num[tr])
    oh=OneHotEncoder(handle_unknown='ignore',min_frequency=100,sparse_output=False).fit(cat.iloc[tr])
    x=np.column_stack((sc.transform(num),oh.transform(cat))).astype(np.float64)
    x32=x.astype(np.float32)
    # Same explanation panel and categorical references as repaired_cohort_study.py.
    typical=cat.iloc[tr].mode(dropna=False).iloc[0]
    ref=np.r_[np.median(x[tr,:len(FEATURES)],axis=0),
              oh.transform(pd.DataFrame([typical],columns=CATEGORICAL))[0]]
    names=oh.get_feature_names_out(CATEGORICAL)
    groups=[np.asarray([i]) for i in range(len(FEATURES))]
    for c in CATEGORICAL:
        cols=np.flatnonzero(np.array([z.startswith(c+'_') for z in names]))+len(FEATURES)
        groups.append(cols)
    assert sorted(np.concatenate(groups).tolist())==list(range(x.shape[1]))
    gen=np.random.default_rng(761)
    expl=np.r_[gen.choice(te[y[te]==1],128,replace=False),
               gen.choice(te[y[te]==0],128,replace=False)]
    rand_top=np.stack([gen.choice(len(groups),3,replace=False) for _ in expl])
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    results=[]
    for seed in (311,312,313):
        model=MLPClassifier(hidden_layer_sizes=(32,16),alpha=.1,max_iter=args.max_iter,
            n_iter_no_change=12,random_state=seed).fit(x[tr],y[tr])
        fpath=out/f'native_strong_fp32_{seed}.onnx'
        om=to_onnx(model,x32[:1],target_opset=17,options={id(model):{'zipmap':False}})
        onnx.save(om,fpath)
        inp=om.graph.input[0].name
        hidden_names=('next_activations','next_activations1')
        relus={n.output[0] for n in om.graph.node if n.op_type=='Relu'}
        if not set(hidden_names).issubset(relus):
            raise RuntimeError('FP32 hidden outputs are not post-ReLU tensors')
        probe=copy.deepcopy(om)
        for name,width in zip(hidden_names,(32,16)):
            probe.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[None,width]))
        probe_path=out/f'native_strong_probe_{seed}.onnx'
        onnx.save(probe,probe_path)
        probe_session=ort.InferenceSession(str(probe_path),sess_options=opts,providers=['CPUExecutionProvider'])
        f=ort.InferenceSession(str(fpath),sess_options=opts,providers=['CPUExecutionProvider'])
        fp=f.run(None,{inp:x32[te]})[-1][:,1]
        fp_predict=lambda z:f.run(['probabilities'],{inp:np.asarray(z,dtype=np.float32)})[0][:,1]
        fp_attr=grouped_attributions(fp_predict,x[expl],ref,groups)
        threshold=float(np.quantile(fp_predict(x[va])[y[va]==1],.30))
        for size in (64,256):
            rng=np.random.default_rng(90000+seed+size)
            for draw in range(args.draws):
                ids=rng.choice(len(cal),2*size,replace=False)
                member=ids[:size]
                for rule,method in [('minmax',CalibrationMethod.MinMax),('percentile',CalibrationMethod.Percentile)]:
                    qpath=out/f'native_strong_int8_{seed}_{size}_{draw}_{rule}.onnx'
                    extra={'ActivationSymmetric':True,'WeightSymmetric':True}
                    if args.matmul_only:
                        extra['OpTypesToExcludeOutputQuantization']=['MatMul']
                    if rule=='percentile':
                        extra['CalibPercentile']=99.5
                    quantize_static(str(fpath),str(qpath),Rows(inp,x32[cal[member]]),
                        quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,calibrate_method=method,
                        op_types_to_quantize=(['MatMul'] if args.matmul_only else
                                              ['MatMul','Add','Relu'] if args.linear_only else None),
                        extra_options=extra)
                    qm=onnx.load(qpath)
                    const={z.name:numpy_helper.to_array(z) for z in qm.graph.initializer}
                    cast={z.output[0] for z in qm.graph.node
                          if z.op_type=='Cast' and z.input[0]==inp}
                    sources=(next(iter(cast)),)+hidden_names
                    activation_qnodes=[z for z in qm.graph.node if z.op_type=='QuantizeLinear']
                    if args.matmul_only and {z.input[0] for z in activation_qnodes}!=set(sources):
                        raise RuntimeError('MatMul-only model has unexpected activation quantizers: '
                                           +str([z.input[0] for z in activation_qnodes]))
                    nodes=[]
                    for source in sources:
                        matching=[z for z in qm.graph.node
                                  if z.op_type=='QuantizeLinear' and z.input[0]==source]
                        if len(matching)!=1:
                            raise RuntimeError(f'Expected one quantizer for {source}, found {len(matching)}')
                        nodes.append(matching[0])
                    # Hidden quantizers must sit directly on post-ReLU outputs.
                    q_relus={z.output[0] for z in qm.graph.node if z.op_type=='Relu'}
                    if not set(hidden_names).issubset(q_relus):
                        raise RuntimeError('Native hidden quantizers are not post-ReLU')
                    steps=[float(const[z.input[1]].ravel()[0]) for z in nodes]
                    zeros=[int(const[z.input[2]].ravel()[0]) for z in nodes]
                    if any(zero != 0 for zero in zeros):
                        raise RuntimeError(f'Expected symmetric zero points, got {zeros}')
                    activations=[x32[cal[ids]]]+probe_session.run(list(hidden_names),{inp:x32[cal[ids]]})
                    layer_hits=[np.isclose(np.abs(h),step*127,rtol=1e-6,atol=1e-7).any(axis=1)
                                for h,step in zip(activations,steps)]
                    hits=layer_hits[0]
                    all_hits=np.any(np.stack(layer_hits,axis=1),axis=1)
                    q=ort.InferenceSession(str(qpath),sess_options=opts,providers=['CPUExecutionProvider'])
                    qp=q.run(None,{inp:x32[te]})[-1][:,1]
                    explanation={}
                    if size==256:
                        q_predict=lambda z:q.run(['probabilities'],{inp:np.asarray(z,dtype=np.float32)})[0][:,1]
                        qa=grouped_attributions(q_predict,x[expl],ref,groups)
                        explanation=explain_stats(fp_attr,qa,q_predict,x[expl],ref,rand_top,groups)
                        explanation['threshold_disagreement']=float(np.mean((qp>=threshold)!=(fp>=threshold)))
                        if rule=='minmax' and args.matmul_only:
                            sim=Quantized(model,x[cal[member]],8,8,False,100)
                            sp=sim.predict(x[te])
                            sa=grouped_attributions(sim.predict,x[expl],ref,groups)
                            stats=explain_stats(fp_attr,sa,sim.predict,x[expl],ref,rand_top,groups)
                            explanation.update({f'simulated_{k}':v for k,v in stats.items()})
                            explanation['simulated_threshold_disagreement']=float(
                                np.mean((sp>=threshold)!=(fp>=threshold)))
                            explanation['simulated_auroc']=float(roc_auc_score(y[te],sp))
                            explanation['native_vs_simulated_mean_absolute_prediction_change']=float(
                                np.mean(np.abs(qp-sp)))
                    results.append(dict(seed=seed,size=size,draw=draw,rule=rule,
                       members_flagged=int(hits[:size].sum()),nonmembers_flagged=int(hits[size:].sum()),
                       members_flagged_three_scales=int(all_hits[:size].sum()),
                       nonmembers_flagged_three_scales=int(all_hits[size:].sum()),
                       members_flagged_hidden1=int(layer_hits[1][:size].sum()),
                       nonmembers_flagged_hidden1=int(layer_hits[1][size:].sum()),
                       members_flagged_hidden2=int(layer_hits[2][:size].sum()),
                       nonmembers_flagged_hidden2=int(layer_hits[2][size:].sum()),
                       count=size,step=steps[0],zero_point=zeros[0],boundary=steps[0]*127,
                       hidden1_step=steps[1],hidden2_step=steps[2],
                       hidden1_zero_point=zeros[1],hidden2_zero_point=zeros[2],
                       hidden_quantizers_post_relu=True,
                       member_boundary_match=bool(hits[:size].any()),
                       fp32_auroc=roc_auc_score(y[te],fp),
                       int8_auroc=roc_auc_score(y[te],qp),
                       mean_absolute_prediction_change=np.mean(np.abs(fp-qp)),
                       unique_fp32_probabilities=int(np.unique(fp).size),
                       unique_int8_probabilities=int(np.unique(qp).size),
                       activation_quantizer_count=len(activation_qnodes),
                       probability_output_scale=float(next((const[z.input[1]].ravel()[0]
                          for z in activation_qnodes if z.input[0].startswith('probabilities')),np.nan)),
                       **explanation))
                    qpath.unlink()
                print('native',seed,size,draw,flush=True)
        probe_path.unlink()
    df=pd.DataFrame(results)
    df.to_csv(out/'native_stronger_onnx_trials.csv',index=False)
    df[df['size']==256].groupby('rule')[
        ['top3_overlap_eligible','eligible_fraction','attr_l1_absolute_mean',
         'fp_attr_l1_absolute_mean','top3_deletion','random3_deletion',
         'threshold_disagreement','int8_auroc']].agg(['mean','min','max']).to_csv(
             out/'native_int8_explanation_summary.csv')
    print(pd.DataFrame(results).groupby(['size','rule'])[
        ['members_flagged','nonmembers_flagged','members_flagged_three_scales',
         'nonmembers_flagged_three_scales','count','int8_auroc']].agg({
        'members_flagged':'sum','nonmembers_flagged':'sum',
        'members_flagged_three_scales':'sum','nonmembers_flagged_three_scales':'sum',
        'count':'sum','int8_auroc':'mean'}).to_string())

if __name__=='__main__':
    main()
