//! Hidden Markov models: the occasionally dishonest casino, decoded with Viterbi and learned by Baum-Welch.

use lib_jsl_machine_learning::supervised::classification::HiddenMarkovModel;
use ndarray::arr2;

fn main() {
    // State 0 is a fair die, state 1 a loaded one that shows six half the time.
    let fair = [1.0 / 6.0; 6];
    let loaded = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5];
    let casino = HiddenMarkovModel::new(vec![0.5, 0.5], arr2(&[[0.95, 0.05], [0.1, 0.9]]), arr2(&[fair, loaded])).unwrap();
    let (states, rolls) = casino.sample(60, 4);
    let (path, log_p) = casino.viterbi(&rolls).unwrap();
    let show = |v: &[usize]| v.iter().map(|s| s.to_string()).collect::<String>();
    println!("rolls   {}", rolls.iter().map(|r| (r + 1).to_string()).collect::<String>());
    println!("truth   {}", show(&states));
    println!("Viterbi {}  (log probability {log_p:.1})", show(&path));
    println!("log-likelihood of the rolls {:.1}", casino.log_likelihood(&rolls).unwrap());

    // Learn the parameters back from 20 000 rolls, starting from a random model.
    let sequences: Vec<Vec<usize>> = (0..40).map(|k| casino.sample(500, 100 + k).1).collect();
    let fit = HiddenMarkovModel::random(2, 6, 2).unwrap().baum_welch(&sequences, 2000, 1e-9).unwrap();
    println!("Baum-Welch: {} iterations, converged {}", fit.log_likelihood.len(), fit.converged);
    println!("transition\n{:.3}\nemission\n{:.3}", fit.model.transition(), fit.model.emission());
}
