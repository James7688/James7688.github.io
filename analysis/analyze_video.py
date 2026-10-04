#!/usr/bin/env python3
import argparse, json, math, os
from pathlib import Path

import cv2
import numpy as np

def ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_sheet(images, labels, out, cols=4, thumb_w=480):
    if not images:
        return
    thumbs=[]
    for im,label in zip(images,labels):
        h,w=im.shape[:2]
        scale=thumb_w/w
        th=cv2.resize(im,(thumb_w,max(1,int(h*scale))),interpolation=cv2.INTER_AREA)
        cv2.rectangle(th,(0,0),(thumb_w,32),(0,0,0),-1)
        cv2.putText(th,label,(8,23),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
        thumbs.append(th)
    cell_h=max(x.shape[0] for x in thumbs)
    rows=math.ceil(len(thumbs)/cols)
    sheet=np.zeros((rows*cell_h,cols*thumb_w,3),dtype=np.uint8)
    for i,th in enumerate(thumbs):
        y=(i//cols)*cell_h; x=(i%cols)*thumb_w
        sheet[y:y+th.shape[0],x:x+th.shape[1]]=th
    cv2.imwrite(str(out),sheet)

def frame_at(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
    ok,frame=cap.read()
    return frame if ok else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out",default="analysis-output")
    args=ap.parse_args()

    out=Path(args.out)
    full=out/"contact-sheets"
    keys=out/"keystrokes"
    events=out/"events"
    for p in (out,full,keys,events): ensure(p)

    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open video")
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration=(n/fps) if fps else 0

    meta={"fps":fps,"frames":n,"width":w,"height":h,"duration_sec":duration}

    # 1s whole-fight contact sheets, split into manageable pages.
    times=np.arange(0,duration+1e-9,1.0)
    imgs=[]; labs=[]; page=1
    for t in times:
        fr=frame_at(cap,fps,float(t))
        if fr is None: continue
        imgs.append(fr); labs.append(f"{t:6.2f}s")
        if len(imgs)==20:
            save_sheet(imgs,labs,full/f"fight_{page:02d}.jpg",cols=4,thumb_w=420)
            imgs=[]; labs=[]; page+=1
    if imgs: save_sheet(imgs,labs,full/f"fight_{page:02d}.jpg",cols=4,thumb_w=420)

    # High-frequency keystroke-area strips. We deliberately save a broad
    # bottom-left crop because overlay layouts differ between clients.
    key_times=np.arange(0,duration+1e-9,0.10)
    kimgs=[]; klabs=[]; kpage=1
    kx0,kx1=0,max(1,int(w*0.36))
    ky0,ky1=max(0,int(h*0.48)),h
    prev_gray=None
    motion=[]
    cap.set(cv2.CAP_PROP_POS_MSEC,0)
    for t in key_times:
        fr=frame_at(cap,fps,float(t))
        if fr is None: continue
        crop=fr[ky0:ky1,kx0:kx1].copy()
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and gray.shape==prev_gray.shape:
            diff=float(np.mean(cv2.absdiff(gray,prev_gray)))
            motion.append({"time_sec":round(float(t),3),"crop_change":round(diff,4)})
        prev_gray=gray
        kimgs.append(crop); klabs.append(f"{t:6.2f}s")
        if len(kimgs)==24:
            save_sheet(kimgs,klabs,keys/f"keys_{kpage:03d}.jpg",cols=6,thumb_w=240)
            kimgs=[]; klabs=[]; kpage+=1
    if kimgs: save_sheet(kimgs,klabs,keys/f"keys_{kpage:03d}.jpg",cols=6,thumb_w=240)

    # Motion peaks in the keystroke crop are candidates for rapid input/UI changes.
    vals=np.array([m["crop_change"] for m in motion],dtype=np.float32)
    threshold=float(np.percentile(vals,92)) if len(vals) else 0.0
    peaks=[]
    last=-999
    for m in motion:
        if m["crop_change"]>=threshold and m["time_sec"]-last>=0.35:
            peaks.append(m); last=m["time_sec"]
    peaks=sorted(peaks,key=lambda x:x["crop_change"],reverse=True)[:40]
    peaks=sorted(peaks,key=lambda x:x["time_sec"])

    # ±0.6s event sheets at 0.15s spacing around motion peaks.
    for i,p in enumerate(peaks,1):
        center=p["time_sec"]
        eimgs=[]; elabs=[]
        for t in np.arange(max(0,center-0.6),min(duration,center+0.6001),0.15):
            fr=frame_at(cap,fps,float(t))
            if fr is not None:
                eimgs.append(fr); elabs.append(f"{t:6.2f}s")
        save_sheet(eimgs,elabs,events/f"event_{i:02d}_{center:07.2f}s.jpg",cols=4,thumb_w=480)

    report={
        "video":str(args.video),
        "metadata":meta,
        "keystroke_crop":{"x0":kx0,"x1":kx1,"y0":ky0,"y1":ky1},
        "rapid_change_threshold":threshold,
        "rapid_change_candidates":peaks,
        "notes":[
            "rapid_change_candidates are visual-change candidates, not definitive key presses",
            "human review of keystroke sheets is required to distinguish WASD changes from HUD/combat animation"
        ],
    }
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(meta,indent=2))
    print(f"wrote {out}")

if __name__=="__main__":
    main()
