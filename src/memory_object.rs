use yarvk::binding_resource::BindingResource;
use yarvk::device_memory::State::Unbound;
use yarvk::device_memory::UnboundResource;

use yarvk::{ContinuousBuffer, ContinuousBufferBuilder, ContinuousImage, ContinuousImageBuilder};

pub type MemoryResource<T> = dyn MemoryBackedResource<RawTy = T>;

pub trait MemoryBackedResource: BindingResource {
    fn memory_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result>;
}

pub trait MemoryObjectBuilder {
    type Ty: UnboundResource + 'static;
    fn build(&self) -> Result<Self::Ty, yarvk::Result>;
}

impl MemoryObjectBuilder for ContinuousBufferBuilder {
    type Ty = ContinuousBuffer<{ Unbound }>;

    fn build(&self) -> Result<Self::Ty, yarvk::Result> {
        self.build()
    }
}

impl MemoryObjectBuilder for ContinuousImageBuilder {
    type Ty = ContinuousImage<{ Unbound }>;

    fn build(&self) -> Result<Self::Ty, yarvk::Result> {
        self.build()
    }
}
