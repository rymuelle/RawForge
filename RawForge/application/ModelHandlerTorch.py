import torch
from pathlib import Path
from platformdirs import user_data_dir
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from RawForge.application.MODEL_REGISTRY import MODEL_REGISTRY
from RawForge.application.helpers.torchutils import can_use_gpu

key_string = """-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEA8iRGMPqFIFVF0TM/AbMI
DJUqdjY1S7dGn6rYjLixnhohHLKIo2ZhFUfPaeYrDoqJblP9MxbBLm6a782/Us0A
vblTQOsdVFHOlVEiDUkG9CJrzh7arqJF+v2LLP9qPIcL5QdIHM+BCKlbNPBU/TJB
49b6a+1FfKCEeY1z9F8H6GCHGeRB43lz5/1yMoBnq//Rc7NrvinwlNcFYHHM1oj6
Hk6KPkgitya11QgTTva+XimR7cbw7h9/vJKbrS7tValApio3Ypmx7AKf6/k16S9K
BCFDN3cyWmjItQNzEWbO2nuM9d3PX2O4FcZVfsA/GU0qSuKFUrrN0KcxKGglLdu4
3Nt3JmOh+VebVWPSTeMzn2R1LDs2CsDpGG+KnHso80HBBq6RuHTugTiUZ2EwjiXN
lRS7olKFQOPwT0tm1EVkH8IxQgV4KJbCb6hAScvWfsDdsP+bu4R+QI9hfU6HCWG3
a8w1AY+5GT7zp1pzKifmnXgMXF3VnAPTGRhpIvPQfum2+tppLZueXlalobK0MDzi
n36TNhRELao1W7Tvc18fxyZn37BBgKs89JO85/cjD72yhVowW7Hy9lL7RnB+etaN
ehXoYFsJReNmD5KNgRtmXbsCUJ+D8v7BVYNGl1UgebmQnMdMWyiU/3l1Uuy8HS3L
1QJYp42f5QqONttCqVzgzrECAwEAAQ==
-----END PUBLIC KEY-----"""


class ModelHandlerTorch:
    """
    Manages the LifeCycle of the Model, the RawHandler, and the Worker Thread.
    """

    def __init__(self, verbose=1):
        super().__init__()
        self.verbose = verbose
        self.model = None
        # Manage devices
        devices = {
            "cuda": can_use_gpu(),
            "mps": torch.backends.mps.is_available(),
            "cpu": lambda: True,
        }
        self.devices = [d for d, is_available in devices.items() if is_available]
        self.set_device(self.devices[0])

        self.filename = None
        self.start_time = None
        self.model_params = {}

        self.pub = serialization.load_pem_public_key(key_string.encode("utf-8"))

    def load_model(self, model_key):
        """Loads a model by key from the registry"""
        if model_key not in MODEL_REGISTRY:
            if self.verbose > 0:
                print(f"Model {model_key} not found in registry.")
            return

        conf = MODEL_REGISTRY[model_key]
        self.model_params = conf
        app_name = "RawForge"
        data_dir = Path(user_data_dir(app_name))
        model_path = data_dir / conf["torchfilename"]

        # Handle Download
        if not model_path.is_file():
            if conf["torchurl"]:
                if self.verbose > 0:
                    print(f"Downloading {model_key}...")
                if not self._download_file(conf["torchurl"], model_path):
                    if self.verbose > 0:
                        print("Failed to download model.")
                    return
            else:
                if self.verbose > 0:
                    print(f"Model file not found at {model_path}")
                return

        try:
            if self.verbose > 1:
                print(f"Loading model: {model_path}")
            # Verify model before load
            self._verify_model(
                model_path, model_path.with_suffix(f"{model_path.suffix}.sig")
            )

            loaded = torch.jit.load(model_path, map_location="cpu")
            self.model = loaded.eval().to(self.device)
        except Exception as e:
            print(f"Failed to load model: {e}")

    def set_device(self, device):
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)
        if self.verbose > 1:
            print(f"Using Device {self.device} from {device}")

    def _verify_model(self, dest_path, sig_path):
        try:
            data = Path(dest_path).read_bytes()
            signature = Path(sig_path).read_bytes()
            self.pub.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            if self.verbose > 1:
                print(f"Model {dest_path} verified!")
            return True
        except Exception as e:
            print(e)
            if dest_path.exists():
                dest_path.unlink()
            if sig_path.exists():
                sig_path.unlink()
            if self.verbose > 0:
                print(f"Model {dest_path} not verified! Deleting.")
            return False

    def _download_file(self, url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Download model signature
            r = requests.get(url + ".sig", stream=True)
            r.raise_for_status()
            sig_path = dest_path.with_suffix(f"{dest_path.suffix}.sig")
            with open(sig_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return self._verify_model(dest_path, sig_path)

        except Exception as e:
            print(e)
            return False
