import habx_features as features


class Features:
    @classmethod
    def do_door(cls) -> bool:
        return features.get_flag('optimizer-v2.do_door')

    @classmethod
    def disable_error_reporting(cls) -> bool:
        return features.get_flag('optimizer-v2.disable_error_reporting')
