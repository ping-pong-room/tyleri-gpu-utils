use crate::memory_object::{MemoryBackedResource, MemoryObjectBuilder, MemoryResource};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::sync::Arc;
use yarvk::binding_resource::BindingResource;
use yarvk::device::Device;
use yarvk::device_memory::{DeviceMemory, IMemoryRequirements, UnboundResource};
use yarvk::physical_device::memory_properties::MemoryType;
use yarvk::DeviceSize;
use yarvk::WHOLE_SIZE;

struct ArrayMemoryObject<T: UnboundResource> {
    object: T::BoundType,
    offset: DeviceSize,
    device_memory: Arc<DeviceMemory>,
}

impl<T: UnboundResource> BindingResource for ArrayMemoryObject<T> {
    type RawTy = T::RawTy;

    fn raw(&self) -> &Self::RawTy {
        self.object.raw()
    }

    fn raw_mut(&mut self) -> &mut Self::RawTy {
        self.object.raw_mut()
    }
}

impl<T: UnboundResource> MemoryBackedResource for ArrayMemoryObject<T> {
    fn memory_memory(&mut self, f: &dyn Fn(&mut [u8])) -> Result<(), yarvk::Result> {
        let size = self.object.get_memory_requirements().size;
        let memory = self.device_memory.get_memory(self.offset, size)?;
        f(memory);
        Ok(())
    }
}

pub struct ArrayDeviceMemory {}
impl ArrayDeviceMemory {
    pub fn new_with_built<
        T: UnboundResource + 'static,
        Builder: Send + Sync + MemoryObjectBuilder<Ty = T>,
    >(
        device: &Arc<Device>,
        object_builder: &Builder,
        built: T,
        counts: usize,
        memory_type: &MemoryType,
    ) -> Result<Vec<Arc<MemoryResource<T::RawTy>>>, yarvk::Result> {
        let mut results: Vec<Arc<MemoryResource<T::RawTy>>> = Vec::with_capacity(counts);
        let unbound = built;
        let memory_requirements = unbound.get_memory_requirements();
        let padding_size =
            memory_requirements.size + memory_requirements.size % memory_requirements.alignment;
        let mut device_memory = DeviceMemory::builder(memory_type, device.clone())
            .allocation_size(padding_size * counts as u64)
            .build()?;
        let _map_result = device_memory.map_memory(0, WHOLE_SIZE);
        let device_memory = Arc::new(device_memory);
        let bound = unbound.bind_memory(&device_memory, 0)?;
        let bound = Arc::new(ArrayMemoryObject::<T> {
            object: bound,
            offset: 0,
            device_memory: device_memory.clone(),
        });
        results.push(bound);
        (1..counts)
            .into_par_iter()
            .map(|i| {
                let unbound = object_builder.build().unwrap();
                let offset = i as DeviceSize * padding_size;
                let bound = unbound.bind_memory(&device_memory, offset).unwrap();
                Arc::new(ArrayMemoryObject::<T> {
                    object: bound,
                    offset,
                    device_memory: device_memory.clone(),
                }) as _
            })
            .collect_into_vec(&mut results);
        Ok(results)
    }
    // pub fn new<T: MemoryObjectBuilder + Send + Sync>(
    //     device: &Arc<Device>,
    //     object_builder: &T,
    //     counts: usize,
    //     memory_type: &MemoryType,
    // ) -> Result<Vec<T::MemoryObjectTy>, yarvk::Result> {
    //     let unbound = object_builder.build()?;
    //     Self::new_with_built(device, object_builder, unbound, counts, memory_type)
    // }
}
