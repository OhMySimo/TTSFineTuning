Monitoring a 4-GPU training run is like flying a plane: you have your **Dashboard** (Terminal), your **Flight Computer** (TensorBoard), and your **Engine Gauges** (System Resources).

Here is your complete monitoring workflow, ranked from "Quick Check" to "Deep Analysis."

### 1. The "Cockpit" View (Terminal + Tmux)

**Use this for:** *Real-time status and ensuring the process doesn't die if your internet disconnects.*

Since you are likely on a remote server (Vast.ai), **never** run the training directly in your SSH session. If your WiFi blips, the training stops.

* **Step 1: Start a Session**
```bash
tmux new -s training

```


* **Step 2: Launch Training**
Run your `./train_vast_round2.sh` script here.
* **Step 3: Detach (Safe Mode)**
Press `Ctrl+B`, then release and press `D`.
* You are now back in your main shell. The training is running safely in the background.


* **Step 4: Re-attach (Check In)**
To see the scrolling logs again:
```bash
tmux attach -t training

```



---

### 2. The "Quick Vitals" Check (Your Custom Script)

**Use this for:** *A 5-second health check without opening complex graphs.*

You uploaded `check_progress.py`. This script is designed to read the event logs and give you a summary.

* **Workflow:**
Open a *new* terminal window (or split pane) and run:
```bash
# You might need to specify the folder if it's not hardcoded
# Assuming the script defaults to current dir or you edit the path inside
python3 check_progress.py

```


*(Note: If the script complains about missing paths, edit the `log_dir` variable inside `check_progress.py` to point to `"output_round2_complete"`)*
* **What to look for in the output:**
* **Gap (Val - Train):** If this > 0.8, you are overfitting (memorizing data).
* **Val Loss:** Should be trending **down**. If it goes UP for 3 checks in a row, early stopping might trigger soon.



---

### 3. The "Deep Dive" (TensorBoard)

**Use this for:** *Visualizing trends, debugging spikes, and checking speaker adaptation.*

This is the most important tool. It plots the curves so you can see *history*.

**How to set it up:**

**Option A: Local Forwarding (Recommended)**
If you want to view the graphs on your own laptop's browser:

1. **On your Laptop (Terminal):**
```bash
# Replace root@IP and PORT with your Vast.ai details
ssh -L 6006:localhost:6006 -p <PORT> root@<IP>

```


2. **On the Server (SSH Session):**
```bash
tensorboard --logdir output_round2_complete --port 6006 --bind_all

```


3. **On your Laptop (Browser):**
Go to `http://localhost:6006`

**What to Monitor:**

* **`loss/total`:** The main curve. Should look like a smooth "hockey stick" (steep drop, then slow decay).
* **`loss/duration`:** The duration predictor. If this stays flat, the model will speak with weird timing.
* **`grad_norm`:** This measures stability.
* *Steady (0.3 - 0.6):* Good.
* *Spikes (> 10.0):* Instability (bad batch or learning rate too high).


* **`speaker_encoder_loss`:** (If available) Shows if the voice cloning is adapting.

---

### 4. The "Engine Room" (System Resources)

**Use this for:** *Verifying hardware efficiency.*

Run this in a separate terminal:

```bash
watch -n 1 nvidia-smi

```

* **GPU-Util:** Should be consistently **>90%**.
* *If it drops to 0% often:* Your CPU isn't feeding data fast enough (dataloader bottleneck).


* **Memory-Usage:** Should be high but stable.
* *If it keeps growing:* You have a memory leak (restart training).



---

### 5. Summary: Your Daily Routine

1. **Morning:**
* `ssh` into server.
* `tmux attach -t training`  Is it still running? Any errors?
* `Ctrl+B, D` to detach.


2. **Afternoon Check:**
* Run `python3 check_progress.py`.
* Is Validation Loss decreasing?
* Is the "Gap" staying small (< 0.5)?


3. **Evening Analysis:**
* Open TensorBoard tunnel.
* Check `grad_norm` for spikes.
* Listen to a generated audio sample if you set up an inference test script (optional but recommended).



**Ready? Launch your `tmux` session and start the script!**
