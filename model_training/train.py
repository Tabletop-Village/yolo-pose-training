from ultralytics import YOLO

# yolo26n-pose.pt auto-downloads from ultralytics' release assets on first
# use. Swap for yolo26s/m/l-pose.pt for a larger model if accuracy needs it.
model = YOLO("yolo26n-pose.pt")

if __name__ == "__main__":
    # optimizer=auto (the default) picks ultralytics' new "Muon"/MuSGD
    # optimizer on this ultralytics version (8.4.137), whose muon_update()
    # crashes with "view size is not compatible with input tensor's size
    # and stride ... Use .reshape(...) instead." on a non-contiguous
    # gradient tensor -- a real bug in that optimizer's implementation,
    # not anything dataset/model-specific. AdamW sidesteps it.
    results = model.train(data="cards.yaml", epochs=100, imgsz=640, save_period=1, optimizer="AdamW")
    model.val()
