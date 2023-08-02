


from typing import Any


class Register:
    def __init__(self, registry_name) -> None:
        self._dict = {}
        self._name = registry_name
    

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"注册的值：{value}必须是callable")
        # 注册器的key加入注册器的名字作为前缀
        if key is None:
            # 为了防止key重复，如果key是None，那么就用value的名字作为key
            key = f"{self._name}_" + value.__name__
        else:
            key = f"{self._name}_" + key
        if key in self._dict:
            print(f"警告：{key}已经存在，将被覆盖")
        self._dict[key] = value
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __contains__(self, key):
        return key in self._dict
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()
    
    def __str__(self) -> str:
        return str(self._dict)
    
    @property
    def name(self):
        return self._name
    
    def register(self, target):
        """
        :param target: 如果target是字符串，那么注册器中的key就是target，
                       如果target是函数，那么注册器中的key就是函数的名字
        """
        def add(key, value):
            self[key] = value
            return value
        if callable(target):
            return add(None, target)
        return lambda x: add(target, x)
    
    def __call__(self, target):
        return self.register(target)


config_registry = Register("cfg")
model_registry = Register("model")
trainer_registry = Register("trainer")