from abc import ABC
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Union
import threading

class LoggingHook(ABC):
    def __init__(self) -> None:
        pass

    def log(self, msg: dict, category: str = None, iter_: int = None) -> None:
        pass

    def format_fixed_width(self, total_width, variable_string):
        if variable_string != "":
            padding = (total_width - len(variable_string) - 2) // 2
            if padding < 0:
                raise ValueError("Variable string is too long for the specified total width.")

            # Build the formatted string
            formatted_string = f"{'=' * padding} {variable_string} {'=' * padding}"

            # Adjust if total_width is odd
            if len(formatted_string) < total_width:
                formatted_string += "="
        else:
            formatted_string = "=" * total_width

        return formatted_string

    def _start_category(self, category: str, total_width : int=60) -> str:
        return self.format_fixed_width(total_width, category)

    def _end_category(self, category: str, total_width: int = 60) -> str:
        return self.format_fixed_width(total_width, "")


class STDOutLoggingHook(LoggingHook):
    def __init__(self, **kwargs) -> None:
        super(STDOutLoggingHook, self).__init__()

    def log(self, msg: dict, category: str = None, iter_: int = None) -> None:
        formatted_msg_items = []
        for k, v in msg.items():
            if isinstance(v, (str)):
                formatted_msg_items.append(f"{k}: {v}")
            else:
                formatted_msg_items.append(f"{k}: {v:.3f}")
        msg = " ".join(formatted_msg_items)
        if iter_:
            msg = f"Iteration {iter_}: {msg}"
        output_length = len(msg)

        if category:
            print(self._start_category(category, output_length))

        print(msg)

        if category:
            print(self._end_category(category, output_length))


class FileLoggingHook(LoggingHook):
    def __init__(self, experiment_dir: str, **kwargs) -> None:
        super(FileLoggingHook, self).__init__()
        self.log_path = os.path.join(experiment_dir, "log.txt")

    def log(self, msg: dict, category: str = None, iter_: int = None) -> None:
        formatted_msg_items = []
        for k, v in msg.items():
            if isinstance(v, (str)):
                formatted_msg_items.append(f"{k}: {v}")
            else:
                formatted_msg_items.append(f"{k}: {v:.3f}")
        msg = " ".join(formatted_msg_items)
        if iter_:
            msg = f"Iteration {iter_}: {msg}"
        output_length = len(msg)

        if category:
            with open(self.log_path, "a") as f:
                f.write(self._start_category(category, output_length) + "\n")

        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

        if category:
            with open(self.log_path, "a") as f:
                f.write(self._end_category(category, output_length) + "\n")


class MetricPlottingHook(LoggingHook):
    def __init__(self, experiment_dir: str, **kwargs) -> None:
        super(MetricPlottingHook, self).__init__()
        self.experiment_dir = os.path.join(experiment_dir, "metrics")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.metrics = {}
        self.current_iter = 0
        self.lock = threading.Lock()

    def log(self, msg: dict, category: str = None, iter_: int = None) -> None:
        if iter_:
            self.current_iter = iter_

        with self.lock:
            for k, v in msg.items():
                key = f"{category}_{k}" if category else k

                if key not in self.metrics:
                    self.metrics[key] = ([], [])
                self.metrics[key][0].append(self.current_iter)
                self.metrics[key][1].append(v)

        # Plot asynchronously on every log call
        threading.Thread(target=self._plot_metrics).start()

    def _plot_metrics(self):
        with self.lock:
            metrics_copy = self.metrics.copy()

        for key, (iters, vals) in metrics_copy.items():
            plt.figure()
            plt.plot(iters, vals)
            plt.title(key)
            plt.xlabel("Iteration")
            plt.ylabel(key)
            plt.savefig(os.path.join(self.experiment_dir, f"{key}.png"))
            plt.close()


logging_hook_factory = {
    "stdout": STDOutLoggingHook,
    "file": FileLoggingHook,
    "metrics": MetricPlottingHook,
}

def build_logging_hook(hooks: Union[str, List[str]], **kwargs) -> List[LoggingHook]:
    if isinstance(hooks, str):
        hooks = [hooks]

    return [logging_hook_factory[hook](**kwargs) for hook in hooks]