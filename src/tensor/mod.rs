pub mod autograd;
pub mod ops;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::tensor::autograd::Node;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

pub fn next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub id: usize,
    pub requires_grad: bool,
    pub node: Option<Arc<Node>>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor { data, shape, id: next_id(), requires_grad: false, node: None }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn grad<'a>(&self, grads: &'a HashMap<usize, Vec<f32>>) -> &'a [f32] {
        grads.get(&self.id).map(|g| g.as_slice()).unwrap_or(&[])
    }

    pub fn detach(&self) -> Tensor {
        Tensor { data: self.data.clone(), shape: self.shape.clone(), id: next_id(), requires_grad: false, node: None }
    }
}
