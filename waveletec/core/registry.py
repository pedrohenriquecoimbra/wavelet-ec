import types


def create_registry(desc=""):
    """Create a new registry dictionary."""
    return regdict({}, __doc__=desc)


class regdict(dict):
    """A dict that supports a custom __doc__ attribute."""

    def __init__(self, *args, __name__=None, __doc__=None, **kwargs):
        super().__init__(*args, **kwargs)
        if __doc__:
            self.__doc__ = __doc__
        if __name__:
            self.__name__ = __name__


def register(name: str, registry, description: str = ""):
    """Decorator to register a function in a given registry."""
    def decorator(func):
        func._correction_name = name
        func._description = description or "No description provided"
        registry[name] = func
        return func
    return decorator


class Register:
    """
    Base class for bricks that require registration.
    """

    def __init__(self):
        # Each instance gets its own registry
        self.registry = create_registry()

    def register(cls, name: str, description: str = ""):
        """Class method that wraps the register decorator for the class's registry."""
        def decorator(func):
            return register(name, cls.registry, description)(func)
        return decorator

    @property
    def available(self):
        return self.registry

    # @classmethod
    def get(self, name):
        """Retrieve a registered method by name.
        Args:
            name (str): The name of the method.
        Raises:
            KeyError: If the method is not registered.
        """
        meth = self.available.get(
            name, None)

        if meth is None:
            raise KeyError(
                f"Method '{name}' is not registered.",
                "Check metadata or make sure the method is registered (if custom method).")
        return meth

    # @classmethod
    def add_method(self, name, how=None):
        """Decorator to add a method dynamically to a class."""
        def decorator(func):
            # Dynamically add the function to the class
            if how == 'classmethod':
                method = classmethod(func)
            elif how == 'static':
                method = func
            else:
                method = types.MethodType(func, self)
            setattr(self, name, method)
            return self
        return decorator

    def apply(self, data, **kwargs):
        """
        Apply the correction routine to the provided data.
        Must be overridden by subclasses.
        """
        raise NotImplementedError(
            "Each correction module must implement an apply() method.")
