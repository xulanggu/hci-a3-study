"""Predictor class copied verbatim from A2 pipeline.py; explicit model path required."""
from pathlib import Path
import numpy as np
from PIL import Image
from config import LABELS, ROOT

def download(*args):
    raise FileNotFoundError("Bundled A2 checkpoint missing; no runtime downloads allowed")

class Predictor:
    """Reusable local inference API; loads the model once for live use."""
    def __init__(self,model=None):
        import onnxruntime as ort
        import onnx
        self.model=Path(model) if model else ROOT/'models/emotion-ferplus-8.onnx'
        if not self.model.exists():download(CFG['model_url'],self.model)
        onnx.checker.check_model(str(self.model))
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(self.model),sess_options=opts,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0]
        if self.input.type!='tensor(float)':raise ValueError(f'Unexpected input dtype: {self.input.type}')
        if self.input.shape[-3:]!=[1,64,64]:raise ValueError(f'Unexpected input shape: {self.input.shape}')
        self.version=ort.__version__

    def predict(self,image_path):
        with Image.open(image_path) as image:
            array=np.asarray(image.convert('L').resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32)[None,None,:,:]
        scores=np.asarray(self.session.run(None,{self.input.name:array})[0],dtype=np.float64).reshape(-1)
        if scores.shape!=(8,) or not np.isfinite(scores).all():raise ValueError('Expected eight finite model scores.')
        scores=np.exp(scores-scores.max());scores/=scores.sum()
        order=np.argsort(-scores,kind='stable')
        return LABELS[order[0]],LABELS[order[1]],scores.tolist()
