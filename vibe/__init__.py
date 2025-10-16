from vibe.trainer import BackdoorTrainer, XConditionedBackdoorTrainer

trainer_factory = {
    "backdoor": BackdoorTrainer,
    "x_conditioned_backdoor": XConditionedBackdoorTrainer,
}