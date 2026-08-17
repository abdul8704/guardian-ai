"""GuardianAI web UI: upload a bench-press video, watch live tracking + triggers.

Run: python app.py   ->  http://localhost:5001
"""
import os
import threading
import time
import queue
import uuid

from flask import (Flask, Response, jsonify, render_template_string, request,
                   redirect, send_file, url_for)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from guardian.pipeline import GuardianPipeline

BASE = os.path.dirname(__file__)
UPLOADS = os.path.join(BASE, "uploads")
OUTPUT = os.path.join(BASE, "output")
_v2 = os.path.join(BASE, "runs_det", "v2", "weights", "best.pt")
WEIGHTS = _v2 if os.path.exists(_v2) else os.path.join(BASE, "runs", "v1", "weights", "best.pt")
OBB_WEIGHTS = os.path.join(BASE, "runs_obb", "v1", "weights", "best.pt")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

app = Flask(__name__)
pipe = None
pipe_lock = threading.Lock()
job_lock = threading.Lock()


class Job:
    def __init__(self, path, name):
        self.id = uuid.uuid4().hex[:8]
        self.path = path
        self.name = name
        self.status = "starting"     # starting | running | done | error
        self.frame = None            # latest annotated jpeg bytes
        self.frames_done = 0
        self.log_lines = []
        self.log_q = queue.Queue()
        self.stop = threading.Event()
        self.out_path = None
        self.error = ""

    def log(self, msg):
        msg = str(msg)
        self.log_lines.append(msg)
        print(msg, flush=True)


current: Job | None = None


def run_job(job: Job):
    global pipe
    import cv2
    try:
        with pipe_lock:
            if pipe is None:
                job.log("[INFO] loading models (first run takes a few seconds) ...")
                pipe = GuardianPipeline(WEIGHTS, obb_weights=OBB_WEIGHTS)
        cap = cv2.VideoCapture(job.path)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        frame_interval = 1.0 / fps_src
        last = [0.0]

        def cb(vis, st, engine):
            if job.stop.is_set():
                job.log("[INFO] job cancelled (replaced by a new upload)")
                return False
            now = time.time()
            wait = frame_interval - (now - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()
            ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                job.frame = jpg.tobytes()
                job.frames_done += 1
                if job.status == "starting":
                    job.status = "running"
            return True

        stem = os.path.splitext(job.name)[0]
        job.out_path = os.path.join(OUTPUT, f"{stem}_tracked.mp4")
        pipe.process(job.path, out_path=job.out_path, frame_cb=cb, log=job.log)
        if not job.stop.is_set():
            job.status = "done"
            job.log(f"[INFO] done. saved: {job.out_path}")
        else:
            job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.log(f"[ALARM] processing error: {e}")


PAGE = """
<!doctype html><title>GuardianAI</title>
<style>
 body{background:#111;color:#ddd;font-family:system-ui,Segoe UI,Arial;margin:0;padding:24px}
 h1{color:#f43;margin:0 0 4px} .sub{color:#888;margin-bottom:20px}
 .row{display:flex;gap:20px;flex-wrap:wrap}
 .panel{background:#1a1a1a;border:1px solid #333;border-radius:10px;padding:16px}
 #videowrap{position:relative;min-width:640px;min-height:360px;display:flex;align-items:center;justify-content:center}
 #video{max-width:860px;width:100%;border-radius:6px;display:none}
 #placeholder{color:#888;font-size:15px;text-align:center}
 .spin{width:28px;height:28px;border:3px solid #333;border-top-color:#f43;border-radius:50%;
       animation:r 0.9s linear infinite;margin:0 auto 12px}
 @keyframes r{to{transform:rotate(360deg)}}
 #log{width:380px;height:520px;overflow-y:auto;font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap}
 .ALARM{color:#ff5252;font-weight:bold} .WARN{color:#ffb300} .INFO{color:#7cb} .RECOVERY{color:#8f8}
 input[type=file]{color:#aaa} button{background:#f43;color:#fff;border:0;padding:8px 18px;border-radius:6px;font-size:15px;cursor:pointer}
 button:disabled{background:#555;cursor:wait}
 #status{margin-left:14px;color:#7cb} a{color:#7cb}
</style>
<h1>GuardianAI</h1><div class=sub>bench-press failure monitoring &mdash; MVP</div>
<div class=panel style="margin-bottom:20px">
 <form id=upform method=post action="/upload" enctype=multipart/form-data style="display:inline">
  <input type=file name=video id=file accept="video/*" required>
  <button type=submit id=upbtn>Upload &amp; monitor</button>
 </form>
 <span id=status></span>
</div>
<div class=row>
 <div class=panel><div id=videowrap>
   <img id=video>
   <div id=placeholder>no video yet &mdash; upload one above</div>
 </div></div>
 <div class=panel><b>Trigger console</b><div id=log></div></div>
</div>
<script>
const $=id=>document.getElementById(id);
let jobId=null, es=null, frameLoop=false, lastUrl=null;

async function pullFrames(id){
  while(frameLoop && jobId===id){
    try{
      const r=await fetch('/frame?job='+id, {cache:'no-store'});
      if(r.status===200){
        const blob=await r.blob();
        const url=URL.createObjectURL(blob);
        $('video').src=url;
        if(lastUrl) URL.revokeObjectURL(lastUrl);
        lastUrl=url;
      } else if(r.status===410){ frameLoop=false; break; }
    }catch(e){}
    await new Promise(res=>setTimeout(res, 66));
  }
}

$('upform').addEventListener('submit', ()=>{ $('upbtn').disabled=true;
  $('status').textContent='uploading...'; });

function attach(id, status){
  jobId=id;
  frameLoop=false;
  $('log').innerHTML='';
  $('video').style.display='none';
  if(status==='done'||status==='error'){
    $('placeholder').textContent='processing already finished — see console / download link above';
  } else {
    $('placeholder').innerHTML='<div class=spin></div>starting analysis ...';
  }
  $('placeholder').style.display='block';
  if(es) es.close();
  es=new EventSource('/events?job='+id);
  es.onmessage=e=>{ if(!e.data) return;
    const d=document.createElement('div');
    const m=e.data.match(/\\[(ALARM|WARN|INFO|RECOVERY)\\]/);
    if(m) d.className=m[1];
    d.textContent=e.data; $('log').appendChild(d);
    $('log').scrollTop=$('log').scrollHeight; };
  es.addEventListener('end', ()=>{ if(es){es.close();es=null;} });
}

async function poll(){
  try{
    const s=await (await fetch('/status')).json();
    if(!s.job){ $('status').textContent=''; return; }
    if(s.job!==jobId) attach(s.job, s.status);
    if(s.status==='running' && !frameLoop){
      $('video').style.display='block';
      $('placeholder').style.display='none';
      frameLoop=true; pullFrames(s.job);
    }
    const labels={starting:'starting ...', running:'analyzing (frame '+s.frames+')',
                  done:'finished', error:'error: '+s.error};
    $('status').textContent = s.name+' — '+labels[s.status];
    if(s.status==='done'){ $('upbtn').disabled=false; frameLoop=false;
      $('status').innerHTML = s.name+' — finished. <a href="/result?job='+s.job+'" download>download tracked video</a>';
      if(es){es.close();es=null;}
    } else if(s.status==='error'){ $('upbtn').disabled=false; frameLoop=false; }
  }catch(e){}
}
setInterval(poll, 700); poll();
</script>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/upload", methods=["POST"])
def upload():
    global current
    f = request.files["video"]
    safe = os.path.basename(f.filename or "upload.mp4")
    path = os.path.join(UPLOADS, f"{uuid.uuid4().hex[:8]}_{safe}")
    f.save(path)
    with job_lock:
        if current is not None and current.status in ("starting", "running"):
            current.stop.set()          # cancel the old job; its thread exits on next frame
        current = Job(path, safe)
        threading.Thread(target=run_job, args=(current,), daemon=True).start()
    return redirect(url_for("index"))


@app.route("/status")
def status():
    j = current
    if j is None:
        return jsonify({"job": None})
    return jsonify({"job": j.id, "name": j.name, "status": j.status,
                    "frames": j.frames_done, "error": j.error})


@app.route("/frame")
def frame():
    j = current
    if j is None or request.args.get("job") != j.id:
        return "gone", 410
    if j.frame is None:
        return "no frame yet", 404
    return Response(j.frame, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/stream")
def stream():
    want = request.args.get("job")
    job = current

    def gen():
        if job is None or job.id != want:
            return
        sent = -1
        while not job.stop.is_set():
            if job.frame is not None and job.frames_done != sent:
                sent = job.frames_done
                yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + job.frame + b"\r\n")
            elif job.status in ("done", "error"):
                break
            time.sleep(0.02)
        # hold the final frame briefly, then end the stream cleanly
        if job.frame is not None:
            yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + job.frame + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/events")
def events():
    want = request.args.get("job")
    job = current
    if job is None or job.id != want:
        return Response("data: \n\n", mimetype="text/event-stream")

    def gen():
        idx = 0
        idle = 0.0
        while True:
            lines = job.log_lines
            if idx < len(lines):
                for line in lines[idx:]:
                    yield f"data: {line}\n\n"
                idx = len(lines)
                idle = 0.0
            elif job.status in ("done", "error") or job.stop.is_set():
                yield "event: end\ndata: \n\n"
                break
            else:
                time.sleep(0.25)
                idle += 0.25
                if idle >= 5.0:
                    idle = 0.0
                    yield ": keepalive\n\n"
    return Response(gen(), mimetype="text/event-stream")


@app.route("/result")
def result():
    j = current
    if j is None or request.args.get("job") != j.id or not j.out_path \
            or not os.path.exists(j.out_path):
        return "no result available", 404
    return send_file(j.out_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, threaded=True)
