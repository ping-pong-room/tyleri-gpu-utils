pub trait ParGroup: IntoIterator {
    type ItemTy;
    fn split_for_par(self) -> Vec<Vec<Self::ItemTy>>;
}

impl<T> ParGroup for Vec<T> {
    type ItemTy = T;

    fn split_for_par(self) -> Vec<Vec<Self::ItemTy>> {
        let current_num_threads = rayon::current_num_threads();
        let size = (self.len() + current_num_threads - 1) / current_num_threads;
        let mut temp = Vec::with_capacity(current_num_threads);
        self.into_iter()
            .filter_map(|t| {
                temp.push(t);
                (temp.len() == size).then(|| {
                    let vec = std::mem::take(&mut temp);
                    temp.reserve(current_num_threads);
                    vec
                })
            })
            .collect()
    }
}
