import typing


@typing.runtime_checkable
class PoolType(typing.Protocol):
    def map(
        self,
        func: typing.Callable[..., typing.Any],
        iterable: typing.Iterable[typing.Any],
        /,
    ) -> typing.Iterable[typing.Any]: ...
