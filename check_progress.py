import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import sys
import time

def analyze_training(log_dir):
    # Trova la cartella dei log più recente
    if not os.path.exists(log_dir):
        print(f"In attesa della cartella {log_dir}...")
        return

    runs_dir = os.path.join(log_dir, "runs")
    if not os.path.exists(runs_dir):
        print("In attesa di log in runs/...")
        return

    # Prendi l'ultima run
    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not subdirs:
        return
    latest_run = max(subdirs, key=os.path.getmtime)

    ea = EventAccumulator(latest_run)
    ea.Reload()

    # Estrai metriche
    try:
        train_loss = ea.Scalars('train/loss_total')
        val_loss = ea.Scalars('eval/loss_total')
    except KeyError:
        print("Dati non ancora disponibili...")
        return

    if not val_loss:
        print("Nessun dato di validazione ancora disponibile.")
        return

    # Prendi l'ultimo valore disponibile
    curr_train = train_loss[-1].value
    curr_val = val_loss[-1].value
    best_val = min([x.value for x in val_loss])
    
    # Calcola il Gap (Generalization Gap)
    gap = curr_val - curr_train

    # Analisi Stato
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*50)
    print(f"📊 MONITORAGGIO TRAINING ROUND 3")
    print("="*50)
    print(f"Step Corrente: {train_loss[-1].step}")
    print("-" * 30)
    print(f"Train Loss:    {curr_train:.4f}")
    print(f"Val Loss:      {curr_val:.4f}")
    print(f"Best Val Loss: {best_val:.4f}")
    print("-" * 30)
    
    print(f"Gap (Val - Train): {gap:.4f}")
    
    # Diagnosi
    print("\nDIAGNOSI:")
    if curr_val < 6.75:
        print("🚀 OBIETTIVO RAGGIUNTO! (< 6.75)")
    elif curr_val > best_val + 0.1:
        print("⚠️  WARNING: OVERFITTING RILEVATO")
        print("    La validation loss sta salendo rispetto al best.")
    elif gap > 0.8:
        print("⚠️  WARNING: High Generalization Gap")
        print("    Il modello impara a memoria il training set.")
    elif curr_val <= best_val:
        print("✅  Tutto ok: Il modello sta migliorando.")
    else:
        print("⚖️  Stallo / Oscillazione normale.")
    print("="*50)

if __name__ == "__main__":
    log_dir = "output_round2_complete"  # matches --output_model_path in launch script
    print(f"Monitorando: {log_dir}")
    while True:
        try:
            analyze_training(log_dir)
            time.sleep(30) # Aggiorna ogni 30 secondi
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Errore lettura: {e}")
            time.sleep(10)