//! https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data

use serde::{Deserialize, Deserializer};
use std::{fs, io::Write, path::PathBuf};
use tracing::info;

/// Converts a string to a boolean.
fn str_to_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    match String::deserialize(deserializer)?.to_lowercase().as_str() {
        "t" | "true" | "1" | "on" | "y" | "yes" => Ok(true),
        "f" | "false" | "0" | "off" | "n" | "no" => Ok(false),
        other => Err(serde::de::Error::invalid_value(
            serde::de::Unexpected::Str(other),
            &"Must be truthy (t, true, 1, on, y, yes) or falsey (f, false, 0, off, n, no)",
        )),
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct Comment {
    #[serde(rename = "CommentId")]
    pub comment_id: String,
    #[serde(rename = "VideoId")]
    pub video_id: String,
    #[serde(rename = "Text")]
    pub text: String,
    #[serde(rename = "IsToxic", deserialize_with = "str_to_bool")]
    pub toxic: bool,
    #[serde(rename = "IsAbusive", deserialize_with = "str_to_bool")]
    pub abusive: bool,
    #[serde(rename = "IsThreat", deserialize_with = "str_to_bool")]
    pub threat: bool,
    #[serde(rename = "IsProvocative", deserialize_with = "str_to_bool")]
    pub provocative: bool,
    #[serde(rename = "IsObscene", deserialize_with = "str_to_bool")]
    pub obscene: bool,
    #[serde(rename = "IsHatespeech", deserialize_with = "str_to_bool")]
    pub hate_speech: bool,
    #[serde(rename = "IsRacist", deserialize_with = "str_to_bool")]
    pub racist: bool,
    #[serde(rename = "IsNationalist", deserialize_with = "str_to_bool")]
    pub nationalist: bool,
    #[serde(rename = "IsSexist", deserialize_with = "str_to_bool")]
    pub sexist: bool,
    #[serde(rename = "IsHomophobic", deserialize_with = "str_to_bool")]
    pub homophobic: bool,
    #[serde(rename = "IsReligiousHate", deserialize_with = "str_to_bool")]
    pub religious_hate: bool,
    #[serde(rename = "IsRadicalism", deserialize_with = "str_to_bool")]
    pub radicalism: bool,
}

pub fn read_csv(path: PathBuf) -> remini_error::Result<Vec<Comment>> {
    let data: Vec<u8> = fs::read(path)?;
    let mut rdr = csv::Reader::from_reader(&data[0..]);

    let mut comments = vec![];
    for result in rdr.deserialize() {
        let record: Comment = result?;
        comments.push(record);
    }

    Ok(comments)
}

pub fn comments_to_txt(comments: Vec<Comment>) -> remini_error::Result<()> {
    info!("Put {} entries on their good directory!", comments.len());

    let categories: Vec<(&str, Box<dyn Fn(&Comment) -> bool>)> = vec![
        (
            "toxic",
            Box::new(|c: &Comment| {
                c.toxic
                    || c.religious_hate
                    || c.provocative
                    || c.abusive
                    || c.nationalist
            }),
        ),
        (
            "severe_toxic",
            Box::new(|c: &Comment| c.hate_speech || c.radicalism || c.racist),
        ),
        ("obscene", Box::new(|c: &Comment| c.obscene)),
        ("threat", Box::new(|c: &Comment| c.threat)),
        (
            "identity_hate",
            Box::new(|c: &Comment| c.homophobic || c.sexist || c.racist),
        ),
        (
            "neutral",
            Box::new(|c: &Comment| {
                !c.toxic
                    && !c.abusive
                    && !c.threat
                    && !c.provocative
                    && !c.obscene
                    && !c.hate_speech
                    && !c.racist
                    && !c.nationalist
                    && !c.sexist
                    && !c.homophobic
                    && !c.religious_hate
                    && !c.radicalism
            }),
        ),
    ];

    for (category, condition) in categories {
        let _ = fs::create_dir_all(format!("./toxicity/{}", category));

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
                .open(format!(
                    "./toxicity/{}/{}-{}.txt",
                    category, comment.video_id, comment.comment_id
                ))?;

            file.write_all(comment.text.as_bytes())?;
        }
    }

    Ok(())
}
