use remini_error::{Error, Result};
use serde::Deserialize;
use std::{fs, io::Write, path::PathBuf};
use tracing::info;

use crate::toxicity::Categorties;

#[derive(Debug, Deserialize, Clone)]
pub struct Comment {
    pub id: String,
    pub comment_text: String,
    pub toxic: u8,
    pub severe_toxic: u8,
    pub obscene: u8,
    pub threat: u8,
    pub insult: u8,
    pub identity_hate: u8,
}

pub fn read_csv(path: PathBuf) -> Result<Vec<Comment>> {
    let data: Vec<u8> = fs::read(path)?;
    let mut rdr = csv::Reader::from_reader(&data[0..]);

    let mut comments = vec![];
    for result in rdr.deserialize() {
        let record: Comment = result.map_err(|_| Error::Unspecified)?;
        comments.push(record);
    }

    Ok(comments)
}

pub fn comments_to_txt(comments: Vec<Comment>) -> Result<()> {
    info!("Put {} entries on their good directory!", comments.len());

    let categories: Categorties<Comment> = vec![
        ("toxic", Box::new(|c: &Comment| c.toxic == 1)),
        ("severe_toxic", Box::new(|c: &Comment| c.severe_toxic == 1)),
        ("obscene", Box::new(|c: &Comment| c.obscene == 1)),
        ("threat", Box::new(|c: &Comment| c.threat == 1)),
        ("insult", Box::new(|c: &Comment| c.insult == 1)),
        (
            "identity_hate",
            Box::new(|c: &Comment| c.identity_hate == 1),
        ),
        (
            "neutral",
            Box::new(|c: &Comment| {
                c.toxic == 0
                    && c.severe_toxic == 0
                    && c.obscene == 0
                    && c.threat == 0
                    && c.insult == 0
                    && c.identity_hate == 0
            }),
        ),
    ];

    for (category, condition) in categories {
        let _ = fs::create_dir_all(format!("./toxicity/{category}"));

        let classified_comments: Vec<&Comment> =
            comments.iter().filter(|c| condition(c)).collect();
        info!(
            "{} entires on {} directory. ({}/{})",
            classified_comments.len(),
            category,
            classified_comments.len(),
            comments.len()
        );

        for comment in classified_comments {
            let mut file = fs::OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(format!("./toxicity/{}/{}.txt", category, comment.id))?;

            file.write_all(comment.comment_text.as_bytes())?;
        }
    }

    Ok(())
}
