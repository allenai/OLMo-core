class Callback:
    """
    Trainer callback base class.
    """

    def pre_train(self):
        """
        Runs before the training loop starts.
        """
        pass

    def post_step(self, step: int):
        """
        Runs after a complete step (potentially including evals and checkpointing).
        """
        del step

    def post_train(self):
        """
        Runs after the training loop is complete.
        """
        pass
