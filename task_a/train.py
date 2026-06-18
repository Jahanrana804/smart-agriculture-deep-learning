import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.dataset import get_dataloaders
from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model, get_model_info, SUPPORTED_MODELS
from utils.metrics import evaluate_model, count_parameters
from utils.plotting import plot_training_curves, plot_confusion_matrix


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(dataloader), correct / total


def train(model_name="custom", aug_intensity="medium", config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}")
    print(f"Model: {model_name} | Aug: {aug_intensity} | Device: {device}")
    print(f"{'='*50}")

    train_dl, val_dl, class_names = get_dataloaders(
        train_dir     = cfg["data"]["train_dir"],
        val_dir       = cfg["data"]["val_dir"],
        batch_size    = cfg["data"]["batch_size"],
        num_workers   = cfg["data"]["num_workers"],
        aug_intensity = aug_intensity,
        image_size    = cfg["data"]["image_size"],
    )
    num_classes = len(class_names)

    if model_name == "custom":
        model = CustomCNN(num_classes)
    else:
        model = get_model(model_name, num_classes)
        get_model_info(model, model_name)

    model = model.to(device)
    count_parameters(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg["train"]["lr"],
        weight_decay = cfg["train"]["weight_decay"],
    )

    if cfg["train"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1
        )

    criterion = nn.CrossEntropyLoss()

    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"],  exist_ok=True)

    best_val_acc = 0.0
    train_losses = []
    val_accs     = []

    for epoch in range(cfg["train"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_dl, optimizer, criterion, device
        )
        metrics  = evaluate_model(model, val_dl, device, num_classes)
        val_acc  = metrics["accuracy"]
        val_f1   = metrics["f1"]

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        scheduler.step()

        print(f"Epoch [{epoch+1:>3}/{cfg['train']['epochs']}] "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(cfg["paths"]["save_dir"], f"{model_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  --> New best ({val_acc:.4f}) saved to {save_path}")

    # save plots
    plot_training_curves(
        train_losses, val_accs,
        save_path=os.path.join(cfg["paths"]["log_dir"], f"{model_name}_curves.png")
    )

    # confusion matrix on final val pass
    metrics = evaluate_model(model, val_dl, device, num_classes)
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        save_path=os.path.join(cfg["paths"]["log_dir"], f"{model_name}_cm.png")
    )

    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    return model, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="custom",
                        help=f"custom | {' | '.join(SUPPORTED_MODELS)}")
    parser.add_argument("--aug",   default="medium",
                        choices=["light", "medium", "heavy"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    train(model_name=args.model, aug_intensity=args.aug, config_path=args.config)
