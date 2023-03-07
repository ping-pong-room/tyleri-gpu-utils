use crate::FxDashMap;
use dashmap::mapref::entry::Entry;
use derive_more::{Deref, DerefMut};
use rayon::iter::ParallelIterator;
use rayon::iter::{IntoParallelIterator, ParallelDrainRange};
use std::sync::Arc;
use std::thread::ThreadId;
use yarvk::command::command_buffer::Level::{PRIMARY, SECONDARY};
use yarvk::command::command_buffer::RenderPassScope::OUTSIDE;
use yarvk::command::command_buffer::State::{EXECUTABLE, INITIAL, INVALID, RECORDING};
use yarvk::command::command_buffer::{
    CommandBuffer, CommandBufferInheritanceInfo, TransientCommandBuffer,
};
use yarvk::device::Device;
use yarvk::fence::{Fence, UnsignaledFence};
use yarvk::physical_device::queue_family_properties::QueueFamilyProperties;
use yarvk::queue::submit_info::{SubmitInfo, Submittable};
use yarvk::queue::Queue;
use yarvk::Handle;

type SecondaryCommandBuffer = CommandBuffer<{ SECONDARY }, { RECORDING }, { OUTSIDE }>;

#[derive(Deref, DerefMut)]
struct ThreadLocalSecondaryBuffer {
    dirty: bool,
    #[deref]
    #[deref_mut]
    buffer: SecondaryCommandBuffer,
}

struct ThreadLocalSecondaryBufferMap {
    queue_family: QueueFamilyProperties,
    device: Arc<Device>,
    command_buffer_inheritance_info: Arc<CommandBufferInheritanceInfo>,
    secondary_buffers: FxDashMap<ThreadId, ThreadLocalSecondaryBuffer>,
    secondary_buffer_handles: FxDashMap<u64, ThreadId>,
}

impl ThreadLocalSecondaryBufferMap {
    fn new(queue: &Queue) -> Self {
        Self {
            queue_family: queue.queue_family_property.clone(),
            device: queue.device.clone(),
            command_buffer_inheritance_info: CommandBufferInheritanceInfo::builder().build(),
            secondary_buffers: Default::default(),
            secondary_buffer_handles: Default::default(),
        }
    }
    fn record_in_thread_local_buffer(
        &self,
        f: impl Fn(&mut SecondaryCommandBuffer) -> Result<(), yarvk::Result>,
    ) -> Result<(), yarvk::Result> {
        let thread_id = std::thread::current().id();
        match self.secondary_buffers.entry(thread_id) {
            Entry::Occupied(mut entry) => {
                let thread_local_buffer = entry.get_mut();
                thread_local_buffer.dirty = true;
                return f(&mut thread_local_buffer.buffer);
            }
            Entry::Vacant(entry) => {
                let secondary_buffer = TransientCommandBuffer::<{ SECONDARY }>::new(
                    &self.device,
                    self.queue_family.clone(),
                )?;
                let mut secondary_buffer =
                    secondary_buffer.begin(self.command_buffer_inheritance_info.clone())?;
                let buffer_handle = secondary_buffer.handle();
                f(&mut secondary_buffer)?;
                let secondary_buffer = ThreadLocalSecondaryBuffer {
                    dirty: true,
                    buffer: secondary_buffer,
                };
                entry.insert(secondary_buffer);
                self.secondary_buffer_handles
                    .insert(buffer_handle, thread_id);
            }
        }
        Ok(())
    }
    fn collect_dirty(&self) -> Vec<CommandBuffer<{ SECONDARY }, { EXECUTABLE }, { OUTSIDE }>> {
        self.secondary_buffers
            .iter()
            .filter(|cb| cb.dirty)
            .map(|cb| *cb.key())
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|thread_id| {
                let (_, cb) = self.secondary_buffers.remove(&thread_id).unwrap();
                cb.buffer.end().unwrap()
            })
            .collect()
    }
    fn return_buffers(
        self: &Arc<Self>,
        buffers: &mut Vec<CommandBuffer<{ SECONDARY }, { INVALID }, { OUTSIDE }>>,
    ) {
        buffers.par_drain(..).for_each(|command_buffer| {
            let command_buffer_inheritance_info = self.command_buffer_inheritance_info.clone();
            let this = self.clone();
            let command_buffer = command_buffer
                .reset()
                .unwrap()
                .begin(command_buffer_inheritance_info)
                .unwrap();
            let thread_id = this
                .secondary_buffer_handles
                .get(&command_buffer.handle())
                .expect("inner error: command handle not exists any more");
            this.secondary_buffers.insert(
                *thread_id,
                ThreadLocalSecondaryBuffer {
                    dirty: false,
                    buffer: command_buffer,
                },
            );
        });
    }
}

#[derive(Deref, DerefMut)]
pub struct ParallelRecordingQueue {
    #[deref]
    #[deref_mut]
    queue: Queue,
    command_buffer: Option<CommandBuffer<{ PRIMARY }, { INITIAL }, { OUTSIDE }>>,
    thread_local_secondary_buffer_map: Arc<ThreadLocalSecondaryBufferMap>,
    fence: Option<UnsignaledFence>,
}

impl ParallelRecordingQueue {
    pub fn new(queue: Queue) -> Result<Self, yarvk::Result> {
        let device = queue.device.clone();
        let command_buffer = TransientCommandBuffer::<{ PRIMARY }>::new(
            &device,
            queue.queue_family_property.clone(),
        )?;
        let fence = Fence::new(&device)?;
        let thread_local_secondary_buffer_map =
            Arc::new(ThreadLocalSecondaryBufferMap::new(&queue));
        Ok(Self {
            queue,
            command_buffer: Some(command_buffer),
            thread_local_secondary_buffer_map,
            fence: Some(fence),
        })
    }
    pub fn record(
        &self,
        f: impl Fn(&mut SecondaryCommandBuffer) -> Result<(), yarvk::Result>,
    ) -> Result<(), yarvk::Result> {
        self.thread_local_secondary_buffer_map
            .record_in_thread_local_buffer(f)
    }
    pub fn simple_submit(&mut self) -> Result<(), yarvk::Result> {
        let command_buffer = self.command_buffer.take().unwrap();
        let command_handle = command_buffer.handle();
        let fence = self.fence.take().unwrap();
        let command_buffer = command_buffer
            .record(|primary_command_buffer| {
                primary_command_buffer.cmd_execute_commands(
                    &mut self.thread_local_secondary_buffer_map.collect_dirty(),
                );
                Ok(())
            })
            .unwrap();
        let submit_info = SubmitInfo::builder()
            .add_one_time_submit_command_buffer(command_buffer)
            .build();
        let fence = Submittable::new()
            .add_submit_info(submit_info)
            .submit(&mut self.queue, fence)?;
        let (signaled_fence, mut submit_result) = fence.wait().unwrap();
        let fence = signaled_fence.reset()?;
        let mut primary_command_buffer = submit_result
            .take_invalid_primary_buffer(&command_handle)
            .unwrap();
        let secondary_buffers = primary_command_buffer.secondary_buffers();
        self.thread_local_secondary_buffer_map
            .return_buffers(secondary_buffers);
        let primary_command_buffer = primary_command_buffer.reset()?;
        self.command_buffer = Some(primary_command_buffer);
        self.fence = Some(fence);
        Ok(())
    }
}
