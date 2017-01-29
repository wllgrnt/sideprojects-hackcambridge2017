#[macro_use]
extern crate csv;

use std::io;
use std::io::prelude::*;
use std::path::Path;
use std::vec::Vec;
use std::fs::File;


fn main() {

      // Split the string, removing the ".pdb" extension and creating a .dat file
  let output = "edgelist.dat";
        // Read the file contents
  let filename = "/home/will/SideProjects/hackcambridge2017/DataFiles/train_PCA_25_whole_set.csv";
  let path = Path::new(&filename);
  let display = path.display();
  let mut rdr = csv::Reader::from_file(path).unwrap().has_headers(true);
  let mut records: Vec<Vec<f32>> = Vec::new();
  // let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();
  'inner: for row in rdr.records() {
      let row = row.unwrap();
      let counter = false;
      let mut record: Vec<f32> = Vec::new();
      for item in row.iter() {
        //   println!("{:?}", item);
        if item.is_empty(){
            continue 'inner;
        }
        else {
            record.push(item.trim().parse::<f32>().unwrap());
        }
      }
      records.push(record);
  }

  let cutoff: f32 = 0.01;
  let output = format!("edgelist{}_PCA_wholeset.dat",cutoff);
  let output = Path::new(&output);
  let display = output.display();
  let f = match File::create(&output) {
      Ok(file) => file,
      Err(e) => panic!("couldn't create {}: {}", display, e)
  };
  let mut writer = io::BufWriter::new(&f);
  // let cutoff: f32 = cutoff;
  for (i, row) in records.iter().enumerate() {
      for (j,row2) in records[0..i].iter().enumerate() {
          let distance_squared: f32 = get_distance_squared(&row, &row2);
        //   println!("{}", distance_squared);
          if distance_squared < cutoff {
              write!(&mut writer, "{} {}\n", i+1, j+1);
          }
      }
  }
 }

fn get_distance_squared(row:&Vec<f32>, row2: &Vec<f32>) -> f32 {

    let mut total:f32 = 0.0;
    for i in 0..row.len() {
        total += (row[i]-row2[i]).powi(2);
    }
    total
}
