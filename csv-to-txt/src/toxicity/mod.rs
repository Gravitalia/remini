pub mod jigsaw;
pub mod youtube;

type Categorties<T> = Vec<(&'static str, Box<dyn Fn(&T) -> bool>)>;
