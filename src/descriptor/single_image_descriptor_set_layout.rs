use std::sync::Arc;

use crate::descriptor::descriptor_pool_list::DescriptorPoolList;
use yarvk::descriptor_set::descriptor_set::DescriptorSetValue;
use yarvk::descriptor_set::descriptor_set_layout::DescriptorSetLayout;
use yarvk::descriptor_set::descriptor_type::DescriptorKind;
use yarvk::descriptor_set::descriptor_variadic_generics::{
    ConstDescriptorSetValue1, DescriptorSetValue1,
};
use yarvk::pipeline::shader_stage::ShaderStage;
use yarvk::sampler::Sampler;

pub type SingleImageDescriptorValue =
    DescriptorSetValue1<0, { DescriptorKind::CombinedImageSamplerImmutable }, 1>;

pub struct SingleImageDescriptorLayout {
    pub desc_set_layout: Arc<DescriptorSetLayout<SingleImageDescriptorValue>>,
    pub descriptor_pool_list: DescriptorPoolList<SingleImageDescriptorValue>,
}

impl SingleImageDescriptorLayout {
    pub fn new(default_sampler: &Arc<Sampler>) -> Self {
        let device = &default_sampler.device;
        let layout_const: <SingleImageDescriptorValue as DescriptorSetValue>::ConstDescriptorSetValue =
            ConstDescriptorSetValue1 {
                t0: ([default_sampler.clone(); 1], ShaderStage::Fragment),
            };

        let desc_set_layout = DescriptorSetLayout::new(&device, layout_const).unwrap();
        let descriptor_pool_list = DescriptorPoolList::new(&desc_set_layout);
        Self {
            desc_set_layout,
            descriptor_pool_list,
        }
    }
}
