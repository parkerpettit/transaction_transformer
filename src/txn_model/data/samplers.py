from typing import List, Iterator
from torch.utils.data import Sampler, RandomSampler, BatchSampler

class AutoBucketSampler(Sampler[List[int]]):
    """
    Buckets examples on the fly by length, then yields batches of size `batch_size`.
    
    Args:
        lengths (List[int]): precomputed sequence-length for each example in the Dataset
        batch_size (int): how many examples per batch
        drop_last (bool): whether to drop the final smaller batch in each bucket
        bucket_size_multiplier (int): proto-bucket size = batch_size * bucket_size_multiplier
    """
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        drop_last: bool = False,
        bucket_size_multiplier: int = 50
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        # how many random indices per proto-bucket
        self.bucket_size = batch_size * bucket_size_multiplier

        # random sampler over all indices
        self.random_sampler = RandomSampler(range(len(lengths)))
        # break that stream into proto-buckets
        self.bucket_sampler = BatchSampler(
            self.random_sampler,
            batch_size=self.bucket_size,
            drop_last=False
        )

    def __iter__(self) -> Iterator[List[int]]:
        for bucket_idxs in self.bucket_sampler:
            # sort proto-bucket by true sequence length
            bucket_idxs.sort(key=lambda i: self.lengths[i])
            # split into final mini-batches
            for i in range(0, len(bucket_idxs), self.batch_size):
                batch = bucket_idxs[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        # approximate number of batches per epoch
        total = sum(len(b) for b in self.bucket_sampler)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size
