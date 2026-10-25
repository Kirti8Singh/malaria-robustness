"""Evaluate the selected ViT checkpoint on the unchanged held-out test split."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from src.vit_data import load_vit_datasets
from src.train_vit import build_model,choose_device,compute_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project-root',default='.'); p.add_argument('--metadata',default=None)
    p.add_argument('--checkpoint',default='results/vit/vit_best.pt'); p.add_argument('--output-dir',default='results/vit')
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto'); a=p.parse_args()
    root=Path(a.project_root).resolve(); out=Path(a.output_dir); out=root/out if not out.is_absolute() else out; out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(a.checkpoint); checkpoint=root/checkpoint if not checkpoint.is_absolute() else checkpoint
    if not checkpoint.is_file(): raise FileNotFoundError(f'Checkpoint not found: {checkpoint}. Train the model first.')
    device=choose_device(a.device); _,datasets=load_vit_datasets(root,a.metadata); ds=datasets['test']
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=='cuda')
    try: ckpt=torch.load(checkpoint,map_location=device,weights_only=True)
    except TypeError: ckpt=torch.load(checkpoint,map_location=device)
    model=build_model(pretrained=False); model.load_state_dict(ckpt['model_state_dict']); model.to(device).eval()
    ys=[]; preds=[]; probs=[]
    with (out/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['sample_id','relative_path','true_label','true_class','predicted_label','predicted_class','probability_parasitized'])
        idx=0
        with torch.no_grad():
            for images,labels in loader:
                logits=model(images.to(device)); batch_probs=torch.softmax(logits,1)[:,1].cpu().tolist(); batch_preds=logits.argmax(1).cpu().tolist(); batch_labels=labels.tolist()
                ys.extend(map(int,batch_labels)); preds.extend(map(int,batch_preds)); probs.extend(map(float,batch_probs))
                for y,pr,prob in zip(batch_labels,batch_preds,batch_probs):
                    path=ds.paths[idx]; writer.writerow([path.name,str(path.relative_to(root)),int(y),'parasitized' if int(y)==1 else 'uninfected',int(pr),'parasitized' if int(pr)==1 else 'uninfected',float(prob)]); idx+=1
    metrics=compute_metrics(ys,preds,probs); metrics.update({'checkpoint':str(checkpoint),'test_samples':len(ds),'positive_class':'parasitized (label 1)','device':str(device),'checkpoint_epoch':ckpt.get('epoch')})
    (out/'test_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    cm=confusion_matrix(ys,preds,labels=[0,1])
    with (out/'test_confusion_matrix.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['true_label','predicted_0_uninfected','predicted_1_parasitized']); w.writerow(['true_0_uninfected',int(cm[0,0]),int(cm[0,1])]); w.writerow(['true_1_parasitized',int(cm[1,0]),int(cm[1,1])])
    print(json.dumps(metrics,indent=2)); print(f'Wrote test outputs to {out}')

if __name__=='__main__': main()
