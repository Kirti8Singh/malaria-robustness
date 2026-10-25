"""Train an ImageNet-pretrained torchvision ViT-B/16 malaria classifier."""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import ViT_B_16_Weights, vit_b_16
from src.vit_data import load_vit_datasets


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()): raise RuntimeError("MPS requested but unavailable")
        return device
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def build_model(pretrained=True):
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    return model


def compute_metrics(labels, preds, probs):
    cm = confusion_matrix(labels, preds, labels=[0, 1]); tn, fp, fn, tp = map(int, cm.ravel())
    return {"accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
            "sensitivity_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None; model.train(training)
    loss_sum = 0.0; count = 0; ys=[]; ps=[]; probs=[]
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, labels in loader:
            images = images.to(device); labels = labels.to(device, dtype=torch.long)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, labels)
            if training: loss.backward(); optimizer.step()
            n = labels.size(0); loss_sum += float(loss.item()) * n; count += n
            ys.extend(labels.detach().cpu().tolist()); ps.extend(logits.argmax(1).detach().cpu().tolist())
            probs.extend(torch.softmax(logits, 1)[:, 1].detach().cpu().tolist())
    if not count: raise RuntimeError("No samples were processed. Check dataset and batch configuration.")
    return loss_sum/count, compute_metrics(ys, ps, probs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', default='.'); p.add_argument('--metadata', default=None)
    p.add_argument('--epochs', type=int, default=10); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-5); p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=0); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    p.add_argument('--output-dir', default='results/vit'); p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--max-train-batches', type=int, default=0, help='Smoke test only; not a final result')
    a=p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.num_workers < 0 or a.max_train_batches < 0: p.error('Invalid training argument values')
    seed_everything(a.seed); device=choose_device(a.device); root=Path(a.project_root).resolve(); out=Path(a.output_dir)
    if not out.is_absolute(): out=root/out
    out.mkdir(parents=True, exist_ok=True)
    metadata, datasets=load_vit_datasets(root, a.metadata)
    kw={'num_workers':a.num_workers, 'pin_memory':device.type=='cuda'}
    train_loader=DataLoader(datasets['train'], batch_size=a.batch_size, shuffle=True, **kw)
    val_loader=DataLoader(datasets['validation'], batch_size=a.batch_size, shuffle=False, **kw)
    model=build_model(not a.no_pretrained).to(device); criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    config={'model':'torchvision ViT-B/16','pretrained':not a.no_pretrained,
      'weights':'ViT_B_16_Weights.DEFAULT' if not a.no_pretrained else None,
      'class_mapping':{'uninfected':0,'parasitized':1},'positive_class':1,
      'epochs_requested':a.epochs,'batch_size':a.batch_size,'learning_rate':a.lr,
      'weight_decay':a.weight_decay,'seed':a.seed,'device':str(device),'num_workers':a.num_workers,
      'metadata_path':str((root/a.metadata).resolve()) if a.metadata and not Path(a.metadata).is_absolute() else (str(Path(a.metadata).resolve()) if a.metadata else str(root/'results/dataset/master_metadata_patient_split_balanced.csv')),
      'split_counts':{k:len(v) for k,v in datasets.items()},'selection_metric':'validation_loss',
      'max_train_batches':a.max_train_batches}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    history=[]; best=float('inf'); best_epoch=0; started=time.time()
    for epoch in range(1,a.epochs+1):
        active=train_loader
        if a.max_train_batches:
            class LimitedLoader:
                def __iter__(self):
                    for i,batch in enumerate(train_loader):
                        if i>=a.max_train_batches: break
                        yield batch
            active=LimitedLoader()
        tl,tm=run_epoch(model,active,criterion,device,optimizer)
        vl,vm=run_epoch(model,val_loader,criterion,device)
        row={'epoch':epoch,'train_loss':tl,'validation_loss':vl}
        row.update({f'train_{k}':v for k,v in tm.items()}); row.update({f'validation_{k}':v for k,v in vm.items()}); history.append(row)
        with (out/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        print(f"Epoch {epoch}/{a.epochs} train_loss={tl:.4f} val_loss={vl:.4f} val_acc={vm['accuracy']:.4f} val_auc={vm['roc_auc']}",flush=True)
        if vl<best:
            best=vl; best_epoch=epoch
            torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'best_validation_loss':best,'config':config,'validation_metrics':vm},out/'vit_best.pt')
    summary={'best_epoch':best_epoch,'best_validation_loss':best,'elapsed_seconds':round(time.time()-started,2),'epochs_completed':len(history),'smoke_test_only':bool(a.max_train_batches)}
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f"Training complete; best checkpoint: {out/'vit_best.pt'}")
    if a.max_train_batches: print('WARNING: smoke test only; do not report these results as the final baseline.')

if __name__=='__main__': main()
