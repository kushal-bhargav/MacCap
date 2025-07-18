from .CapDec import build_model as build_cap_dec
from .DeCap import build_model as build_decap
from .MAGIC import build_model as build_magic
from .MacCap import build_model as build_maccap
from .ROCO import build_model as build_roco  # Adding ROCO model import

def build_model(args):
    if args.model_name == 'MacCap':
        """
        DECODER with cls feature only + noise injected + inject multiple noise during inference
        """
        return build_maccap(args)
    elif args.model_name == 'CAPDEC':
        """
        Text-Only Training for Image Captioning using Noise-Injected CLIP
        """
        return build_cap_dec(args)
    elif args.model_name == 'DECAP':
        """
        DECAP: DECODING CLIP LATENTS FOR ZERO-SHOT CAPTIONING VIA TEXT-ONLY TRAINING
        """
        return build_decap(args)
    elif args.model_name == 'MAGIC':
        """
        Language Models Can See: Plugging Visual Controls in Text Generation
        """
        return build_magic(args)
    elif args.model_name == 'ROCO':  # Adding ROCO model handling
        """
        ROCO: Radiology Objects in Context - Handling medical image captioning
        """
        return build_roco(args)

    raise NotImplementedError("Model name {} not implemented".format(args.model_name))
