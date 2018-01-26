use std::iter::once;
use std::marker::PhantomData;
use std::ops::Range;

use gfx_hal::{Backend, Device};
use gfx_hal::pso::{DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType,
                   DescriptorWrite, ShaderStageFlags};

use relevant::Relevant;
use smallvec::SmallVec;

/// Single descriptor binding.
/// Type and count are constant for each binding type, while binding index and stage flags are
/// specified during binding creation.
pub trait Binding: Copy {
    /// Type of the binding.
    const TY: DescriptorType;

    /// Count of binding.
    const COUNT: usize;

    /// Binding index.
    fn binding(self) -> usize;

    /// Stage flags for binding.
    fn stage(self) -> ShaderStageFlags;
}

/// Uniform non-array binding type.
///
/// ### Type parameters:
///
/// - `T`: type expected by shaders using the binding
#[derive(Derivative)]
#[derivative(Clone, Copy, Debug)]
pub struct Uniform<T> {
    binding: usize,
    stage: ShaderStageFlags,
    pd: PhantomData<fn() -> T>,
}

impl<T> Uniform<T> {
    /// Bind uniform to the given descriptor set.
    ///
    /// ### Parameters:
    ///
    /// - `set`: descriptor set to bind uniform to
    /// - `buffer`: buffer where the uniform is located
    /// - `range`: byte range in the buffer where the uniform is located
    fn bind<'a, 'b, B>(
        self,
        set: &'a B::DescriptorSet,
        buffer: &'b B::Buffer,
        range: Range<u64>,
    ) -> DescriptorSetWrite<'a, 'b, B>
    where
        B: Backend,
    {
        DescriptorSetWrite {
            set,
            binding: self.binding(),
            array_offset: 0,
            write: DescriptorWrite::UniformBuffer(vec![(buffer, range)]),
        }
    }
}

impl<T> Binding for Uniform<T> {
    const TY: DescriptorType = DescriptorType::UniformBuffer;
    const COUNT: usize = 1;

    #[inline(always)]
    fn binding(self) -> usize {
        self.binding
    }

    #[inline(always)]
    fn stage(self) -> ShaderStageFlags {
        self.stage
    }
}

/// Heterogeneous list of descriptor bindings.
/// `()` is an empty list, `(H, T)` is `BindingsLists` where `H: Binding` and `T: BindingsList`.
pub trait BindingsList: Copy {
    /// Fill bindings structures.
    fn fill<E>(self, extend: &mut E)
    where
        E: Extend<DescriptorSetLayoutBinding>;
}

impl BindingsList for () {
    fn fill<E>(self, _extend: &mut E) {}
}

impl<H, T> BindingsList for (H, T)
where
    H: Binding,
    T: BindingsList,
{
    fn fill<E>(self, extend: &mut E)
    where
        E: Extend<DescriptorSetLayoutBinding>,
    {
        extend.extend(once(DescriptorSetLayoutBinding {
            ty: H::TY,
            count: H::COUNT,
            stage_flags: self.0.stage(),
            binding: self.0.binding(),
        }));
        self.1.fill(extend);
    }
}

/// Pipeline layout type-level representation.
///
/// ### Type parameters:
///
/// - `L`: `BindingsList`
#[derive(Copy, Clone)]
pub struct Layout<L> {
    bindings: L,
}

impl Layout<()> {
    /// Create an empty layout.
    pub(crate) fn new() -> Self {
        Layout { bindings: () }
    }
}

impl<L> Layout<L>
where
    L: BindingsList,
{
    /// Add uniform binding to the layout.
    ///
    /// ### Parameters:
    ///
    /// - `binding`: index of the binding.
    /// - `stage`: stage or stage flags.
    pub fn uniform<T, S: Into<ShaderStageFlags>>(
        self,
        binding: usize,
        stage: S,
    ) -> Layout<(Uniform<T>, L)> {
        self.with(Uniform {
            binding,
            stage: stage.into(),
            pd: PhantomData,
        })
    }

    /// Get array of bindings.
    pub(crate) fn bindings(self) -> SmallVec<[DescriptorSetLayoutBinding; 64]> {
        let mut bindings = SmallVec::<[_; 64]>::new();
        self.bindings.fill(&mut bindings);
        bindings
    }

    /// Add binding to the layout.
    fn with<B>(self, binding: B) -> Layout<(B, L)> {
        Layout {
            bindings: (binding, self.bindings),
        }
    }
}

/// Binder can be used to bind bindings to a descriptor set.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
/// - `L`: `BindingsList`
pub struct Binder<'a, B: Backend, L> {
    layout: &'a B::PipelineLayout,
    bindings: L,
}

impl<'a, B, L> Binder<'a, B, L>
where
    B: Backend,
    L: Clone,
{
    pub(crate) fn new(layout: &'a B::PipelineLayout, bindings: Layout<L>) -> Self {
        Binder {
            layout,
            bindings: bindings.bindings,
        }
    }

    /// Specify set to start write bindings.
    pub fn set<'b, 'c>(&self, set: &'b mut B::DescriptorSet) -> SetBinder<'b, 'c, B, L>
    where
        B: Backend,
    {
        SetBinder {
            relevant: Relevant,
            bindings: self.bindings.clone(),
            set,
            writes: SmallVec::new(),
        }
    }

    /// Get pipeline layout
    pub fn layout(&self) -> &B::PipelineLayout {
        &self.layout
    }
}

/// Binder to bind bindings to the contained descriptor set, and update encountered uniforms on the
/// graphics device.
///
/// ### Type parameters:
///
/// - `B`: hal `Backend`
/// - `L`: `BindingsList`
pub struct SetBinder<'a, 'b, B: Backend, L> {
    relevant: Relevant,
    bindings: L,
    set: &'a B::DescriptorSet,
    writes: SmallVec<[DescriptorSetWrite<'a, 'b, B>; 64]>,
}

impl<'a, 'b, B> SetBinder<'a, 'b, B, ()>
where
    B: Backend,
{
    /// Bind all written descriptor bindings.
    pub fn bind(self, device: &B::Device) {
        device.update_descriptor_sets(&self.writes);
    }
}

impl<'a, 'b, B, H, T> SetBinder<'a, 'b, B, (Uniform<H>, T)>
where
    B: Backend,
{
    /// Add uniform binding.
    pub fn uniform(self, buffer: &'b B::Buffer, range: Range<u64>) -> SetBinder<'a, 'b, B, T> {
        let SetBinder {
            relevant,
            bindings: (head, tail),
            set,
            mut writes,
        } = self;

        writes.push(head.bind(set, buffer, range));
        SetBinder {
            relevant,
            bindings: tail,
            set,
            writes,
        }
    }
}

impl<'a, 'b, B, H, T> SetBinder<'a, 'b, B, (H, T)>
where
    B: Backend,
{
    /// Intentionally skip one binding.
    pub fn skip<C>(self) -> SetBinder<'a, 'b, B, T> {
        SetBinder {
            relevant: self.relevant,
            bindings: self.bindings.1,
            set: self.set,
            writes: self.writes,
        }
    }
}
