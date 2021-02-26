class HANException(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        if self.message:
            return 'MyCustomError, {0}'.format(self.message)
        else:
            return 'MyCustomError has been raised'
