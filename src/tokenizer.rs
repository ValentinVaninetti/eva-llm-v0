pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn encode(text: &str) -> Vec<usize> {
        text.as_bytes().iter().map(|&b| b as usize).collect()
    }

    pub fn decode(ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
