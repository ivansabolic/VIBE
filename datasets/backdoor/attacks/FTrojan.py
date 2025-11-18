from datasets.backdoor.backdoor import NumpyPoisonedBackdoor

# Re-export NumpyPoisonedBackdoor as FTrojan for consistency with other attack classes
FTrojan = NumpyPoisonedBackdoor

