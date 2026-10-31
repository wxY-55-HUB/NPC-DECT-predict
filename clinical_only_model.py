from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, brier_score_loss
from .data import load_clinical
from .evaluation import delong_auc_ci, youden_threshold
from .utils import load_config, parse_ki67, patient_date, set_determinism, save_json, write_csv

def build_data(cfg):
    c=load_clinical(cfg).copy()
    c=c.rename(columns={"Age":"age","Sex":"sex","Weight":"weight","Height":"height"})
    c["ExamDate"]=c["PatientID"].map(patient_date)
    c["y"]=(c["Ki67"]>=float(cfg["outcomes"]["primary_cutoff_percent"])/100.0).astype(int)
    for col in ["age","weight","height","T_stage","N_stage","M_stage"]:
        c[col]=pd.to_numeric(c[col],errors="coerce")
    c["sex_binary"]=c["sex"].astype(str).str.upper().map({"M":1,"W":0,"F":0,"MALE":1,"FEMALE":0})
    cols=["age","sex_binary","weight","height","T_stage","N_stage","M_stage"]
    c=c.dropna(subset=["Ki67","ExamDate"]).copy()
    return c,cols

def make_estimator(seed):
    prep=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scale",RobustScaler())])
    return Pipeline([("prep",prep),("model",LogisticRegression(penalty="elasticnet",solver="saga",class_weight="balanced",max_iter=10000,random_state=seed))])

def run(config_path, output_dir="results/clinical_only"):
    cfg=load_config(config_path); seed=int(cfg["project"]["random_seed"]); set_determinism(seed)
    root=Path(cfg["paths"]["project_root"]); out=Path(output_dir); out=out if out.is_absolute() else root/out; out.mkdir(parents=True,exist_ok=True)
    started=time.time(); frame,features=build_data(cfg); cutoff=pd.Timestamp(cfg["project"]["temporal_cutoff"])
    dev=frame[frame.ExamDate<cutoff].copy(); val=frame[frame.ExamDate>=cutoff].copy()
    X=dev[features]; y=dev.y.to_numpy(); splitter=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=seed)
    grid={"model__C":[0.03,0.1,0.3,1.0],"model__l1_ratio":[0.25,0.5,0.75,1.0]}
    oof=[]
    for fold,(tr,te) in enumerate(splitter.split(X,y),1):
        inner=StratifiedKFold(n_splits=4,shuffle=True,random_state=seed+fold)
        search=GridSearchCV(make_estimator(seed+fold),grid,scoring="roc_auc",cv=inner,n_jobs=-1,refit=True)
        search.fit(X.iloc[tr],y[tr]); pred=search.predict_proba(X.iloc[te])[:,1]
        oof.extend({"PatientID":dev.iloc[j].PatientID,"repeat_fold":fold,"y_true":int(y[j]),"probability":float(p)} for j,p in zip(te,pred))
    oof_long=pd.DataFrame(oof); oof_unique=oof_long.groupby("PatientID",as_index=False).agg(y_true=("y_true","first"),probability=("probability","mean"))
    thr=youden_threshold(oof_unique.y_true.to_numpy(),oof_unique.probability.to_numpy())
    final_search=GridSearchCV(make_estimator(seed),grid,scoring="roc_auc",cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=seed),n_jobs=-1,refit=True)
    final_search.fit(X,y); p=final_search.predict_proba(val[features])[:,1]; yy=val.y.to_numpy(); pred=(p>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel(); auc,lo,hi,se=delong_auc_ci(yy,p)
    metrics={"model":"clinical_only_elasticnet_logistic","n_total":len(frame),"development_n":len(dev),"validation_n":len(val),"development_high":int(dev.y.sum()),"development_low":int(len(dev)-dev.y.sum()),"validation_high":int(val.y.sum()),"validation_low":int(len(val)-val.y.sum()),"temporal_auc":auc,"auc_ci_low":lo,"auc_ci_high":hi,"auc_se":se,"threshold_from_development_oof":thr,"accuracy":accuracy_score(yy,pred),"sensitivity":recall_score(yy,pred,zero_division=0),"specificity":tn/(tn+fp) if tn+fp else np.nan,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"brier":brier_score_loss(yy,p),"best_params":final_search.best_params_,"features":features,"cv":"development repeated stratified 5-fold x5; inner stratified 4-fold","random_seed":seed}
    oof_unique.to_csv(out/"clinical_only_development_oof.csv",index=False); oof_long.to_csv(out/"clinical_only_development_oof_repeats.csv",index=False)
    pd.DataFrame({"PatientID":val.PatientID.to_numpy(),"y_true":yy,"probability":p,"threshold":thr,"predicted":pred}).to_csv(out/"clinical_only_temporal_predictions.csv",index=False)
    save_json(out/"clinical_only_metrics.json",metrics); save_json(out/"run_manifest.json",{"analysis_type":"standalone_clinical_only","full_pipeline_called":False,"config":str(Path(config_path).resolve()),"output_dir":str(out),"elapsed_minutes":round((time.time()-started)/60,3),**metrics})
    print(json.dumps(metrics,ensure_ascii=False,indent=2)); return out

def main():
    a=argparse.ArgumentParser(); a.add_argument("--config",type=Path,default=Path("config.yaml")); a.add_argument("--output-dir",default="results/clinical_only"); z=a.parse_args(); run(z.config,z.output_dir)
if __name__=="__main__": main()
