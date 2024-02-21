class EarlyStoppingMultiOutput(tf.keras.callbacks.Callback):
    def __init__(self, monitor: list, patience=10, save_weights=False, mode="max"):
        super(EarlyStoppingMultiOutput, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.save_weights = save_weights
        self.mode = mode
        if mode not in ["max", "min"]:
            raise ValueError('mode must be "max" or "min"')
        
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_monitored_values = [0] * len(monitor) if mode == "max" else [float('inf')] * len(monitor)
        
    def on_train_begin(self, logs=None):
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        stop_training = False
        for i, monitor in enumerate(self.monitor):
            current_val = logs.get(monitor)
            if current_val is None:
                continue

            if self.mode == "max" and current_val > self.best_monitored_values[i]:
                self.best_monitored_values[i] = current_val
                self.wait = 0
                if self.save_weights:
                    self.best_weights = self.model.get_weights()
                    
                    
            elif self.mode == "min" and current_val < self.best_monitored_values[i]:
                self.best_monitored_values[i] = current_val
                self.wait = 0
                if self.save_weights:
                    self.best_weights = self.model.get_weights()
            
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.save_weights and self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    stop_training = True
                    break
        
        if stop_training:
            print(f"\nEarly stopping triggered at epoch {self.stopped_epoch + 1}")
            if self.save_weights:
                print("Restoring model weights to the best observed during training.")
