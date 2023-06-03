class BaseTrainingEnv:
    def __init__(self, setting, *args, **kwargs):
        self.settings: dict = setting

    def update(self, display: bool = False):
        pass
