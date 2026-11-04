//! CSV tables and NumPy .npy arrays.

use lib_jsl_file_io::{NpyArray, NpyData, format_csv, format_csv_f64, npy_decode, npy_encode, parse_csv};
use ndarray::array;

fn main() {
    let text = "city,population,note\nParis,2102650,\"capital, France\"\n\"Saint-Étienne\",172718,\"say \"\"Sainté\"\"\"\n";
    let table = parse_csv(text, ',', true).unwrap();
    println!("header {:?}", table.header);
    for row in &table.rows {
        println!("  {row:?}");
    }
    let population = table.column_f64(table.column_index("population").unwrap()).unwrap();
    println!("total population {}", population.iter().sum::<f64>());
    print!("written back:\n{}", format_csv(&table, ';'));
    print!("numbers (NaN as an empty field):\n{}", format_csv_f64(Some(&["x", "y"]), &[vec![0.1, 1e-300], vec![f64::NAN, -2.5]], ','));

    // .npy arrays carry their element type and shape.
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    let bytes = npy_encode(&NpyArray::from_f64(&a)).unwrap();
    println!(".npy: {} bytes, header {:?}", bytes.len(), String::from_utf8_lossy(&bytes[10..60]).trim_end());
    let back = npy_decode(&bytes).unwrap();
    println!("shape {:?}, identical: {}", back.shape, back.to_f64().unwrap() == a);
    let counts = NpyArray::new(vec![2, 2], NpyData::U16(vec![1, 2, 3, 4])).unwrap();
    println!("uint16 array as f64: {:?}", npy_decode(&npy_encode(&counts).unwrap()).unwrap().to_f64().unwrap().into_raw_vec_and_offset().0);
}
