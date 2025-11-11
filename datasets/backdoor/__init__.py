from datasets.backdoor.backdoor import Backdoor, NumpyPoisonedBackdoor

# Import attack classes explicitly to avoid circular import issues
from datasets.backdoor.attacks.BadNets import BadNets
from datasets.backdoor.attacks.Blend import Blend
from datasets.backdoor.attacks.WaNet import WaNet
from datasets.backdoor.attacks.Adap import Adap
from datasets.backdoor.attacks.LC import LC
from datasets.backdoor.attacks.UBA import UBA
from datasets.backdoor.attacks.FTrojan import FTrojan

backdoor_factory = {
    "badnets": BadNets,
    "blend": Blend,
    "wanet": WaNet,
    "ftrojan": FTrojan,
    "adap_patch": Adap,
    "adap_blend": Adap,
    "labelconsistent": LC,
    "uba-patch": UBA,
    "uba-blend": UBA,
}
