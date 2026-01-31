import os

if __name__ == "__main__":
    cmd = (
        "python3 train.py "
        "--workers 4 "
        "--device 0 "
        "--batch-size 4 "
        "--epochs 15 "
        "--img 640 640 "
        "--data data/seadronessee_tiled_train.yaml "
        "--cfg cfg/training/yolov7.yaml "
        "--weights weights/yolov7.pt "
        "--name seadronessee_tiled_train "
        "--hyp data/hyp.scratch.p5.yaml "
    )

    print("Running command:")
    print(cmd)
    os.system(cmd)
