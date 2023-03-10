from dataclasses import dataclass
from functools import partial
from itertools import groupby, starmap, islice
from operator import itemgetter, truediv, ne
from typing import Sequence, Optional, List, Iterable, Tuple
from statistics import fmean

from numpy import ndarray


def truncate(x, len):
    return islice(x, 0, len)


@dataclass
class ASRErrorCalculator:
    token_list: Sequence[str]
    filter_out_space: bool = False
    normalize_length: bool = True
    blank_id: Optional[int] = None
    space_id: Optional[int] = None
    blank_sym: Optional[str] = None
    space_sym: Optional[str] = None

    def __post_init__(self):
        assert self.blank_id is not None or self.blank_sym is not None
        assert self.space_id is not None or self.space_sym is not None

        if self.blank_id is not None:
            self.blank_sym = self.token_list[self.blank_id]
        else:
            self.blank_id = self.token_list.index(self.blank_sym)

        if self.space_id is not None:
            self.space_sym = self.token_list[self.space_id]
        else:
            self.space_id = self.token_list.index(self.space_sym)

    def ctc_decode(self, x: Iterable[int], length: Optional[int] = None) -> Iterable[int]:
        """
        Args:
            x: array of shape (maxlen,)
            length:

        Returns:

        """
        x = truncate(x, length)
        x = groupby(x)
        x = map(itemgetter(0), x)
        x = filter(partial(ne, self.blank_id), x)
        return x

    def ctc_decode_str(self, x: Iterable[int], length: Optional[int] = None) -> str:
        x = self.ctc_decode(x, length)
        x = self.tokens2str(x)
        return x

    def batch_ctc_decode(self, x: ndarray, lengths: ndarray) -> List[Tuple[int]]:
        """
        Args:
            x: (bs, maxlen)
            lengths: (bs,)

        Returns:

        """
        x = starmap(self.ctc_decode, zip(x, lengths))
        x = map(tuple, x)
        return list(x)

    def batch_ctc_decode_str(self, x: ndarray, lengths: ndarray) -> List[str]:
        """
        Args:
            x: (bs, maxlen)
            lengths: (bs,)

        Returns:

        """
        x = starmap(self.ctc_decode_str, zip(x, lengths))
        return list(x)

    def tokens2str(self, x: Iterable[int], length: Optional[int] = None) -> str:
        """

        Args:
            x:
            length:

        Returns:

        """
        x = truncate(x, length)
        x = map(self.token_list.__getitem__, x)
        x = ''.join(x)
        return x

    def batch_tokens2str(self, x: ndarray, lengths: ndarray) -> List[str]:
        """

        Args:
            x:
            lengths:

        Returns:

        """
        x = starmap(self.tokens2str, zip(x, lengths))
        return list(x)

    def calculate_error_rate(self, predictions: Iterable[str], ground_truths: Iterable[str]) -> float:
        """

        Args:
            predictions:
            ground_truths:

        Returns:

        """
        import editdistance

        dist_len_pair = starmap(lambda pred, gt: (editdistance.eval(pred, gt), len(gt)),
                                zip(predictions, ground_truths))
        if self.normalize_length:
            distances, gt_lengths = zip(*dist_len_pair)
            ret = sum(distances) / sum(gt_lengths)
        else:
            ret = fmean(starmap(truediv, dist_len_pair))
        return ret

    def calculate_cer(self, predictions: Iterable[str], ground_truths: Iterable[str]) -> float:
        """

        Args:
            predictions:
            ground_truths:

        Returns:

        """
        if self.filter_out_space:
            predictions, ground_truths = map(partial(map, lambda s: s.replace(self.space_sym, '')),
                                             (predictions, ground_truths))
        return self.calculate_error_rate(predictions, ground_truths)

    def calculate_wer(self, predictions: Iterable[str], ground_truths: Iterable[str]) -> float:
        """

        Args:
            predictions:
            ground_truths:

        Returns:

        """
        predictions, ground_truths = map(partial(map, lambda s: tuple(filter(bool, s.split(self.space_sym)))),
                                         (predictions, ground_truths))
        return self.calculate_error_rate(predictions, ground_truths)
