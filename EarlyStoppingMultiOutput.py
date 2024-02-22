class Early_stopping_multi_output(tf.keras.callbacks.Callback):
    def __init__(self, monitor: list, patience=10, save_weights=False, mode="max"):
        super(Early_stopping_multi_output, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.save_weights = save_weights
        self.mode = mode
        if mode not in ["max", "min"]:
            raise ValueError('mode must be "max" or "min"')
        self.best_weights = None
        self.wait = [0] * len(monitor)
        self.stopped_epoch = 0
        self.best_monitored_values = [0] * len(monitor) if mode == "max" else [float('inf')] * len(monitor)
        
    def on_train_begin(self, logs=None):
        self.wait = [0] * len(self.monitor)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for i, monitor in enumerate(self.monitor):
            current_value = logs.get(monitor)
            if current_value is None:
                continue
            if self.mode == "max" and current_value > self.best_monitored_values[i]:
                self.best_monitored_values[i] = current_value
                self.wait[i] = 0
                if self.save_weights:
                    self.best_weights = self.model.get_weights()         
            elif self.mode == "min" and current_value < self.best_monitored_values[i]:
                self.best_monitored_values[i] = current_value
                self.wait[i] = 0
                if self.save_weights:
                    self.best_weights = self.model.get_weights()

            else:
                self.wait[i] += 1
        
        if max(self.wait) >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.save_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
            print(f"\nEarly stopping triggered at epoch {self.stopped_epoch + 1}")
            if self.save_weights:
                print("Restoring model weights to the best observed during training.")
