from datasets.backdoor.backdoor import Backdoor, NumpyPoisonedBackdoor
from datasets.backdoor.attacks import *

backdoor_factory = {
    "badnets": BadNets,
    "blend": Blend,
    "wanet": WaNet,
    "ftrojan": NumpyPoisonedBackdoor,
    "adap_patch": Adap,
    "adap_blend": Adap,
    "labelconsistent": LC,
    "uba-patch": UBA,
    "uba-blend": UBA,
}
