/**
* Name: TicTacToe
* Description: The canonical turn-based (AEC) example, ported to GAMA from PettingZoo's
*              `classic/tictactoe_v3` so the two can be checked against each other.
*
* Why this is the reference example for GamaAECEnv:
*   - a board game CANNOT be modelled with simultaneous actions, so no one has to be
*     convinced that the AEC API is the right one here;
*   - perfect play is a known draw, so "did it learn" is measured against ground truth
*     rather than against a hand-written heuristic of unknown quality;
*   - an episode is at most 9 turns, so training takes minutes and needs no neural network;
*   - the rules are deterministic — nothing depends on a seed.
*
* Encoding follows tictactoe_v3: the observation is two 3x3 planes (own marks, then the
* opponent's) flattened to 18 values, the action is a cell index in 0..8, and the legal moves
* are published as an `action_mask` in `infos`.
*
* GAML-side contract expected by GamaAECEnv:
*   PetzAgent[0].observe_one("<agent>") / set_action_one("<agent>", '<json>')
*   PetzAgent[0].end_round() / publish() / data
*/

model TicTacToe

global {

	// 0 = empty, 1 = player_1 (X), 2 = player_2 (O)
	list<int> board <- list_with(9, 0);
	int winner <- 0;			// 0 = none yet, 1 or 2 = that player, 3 = draw
	bool game_over <- false;

	list<string> rl_agents <- ["player_1", "player_2"];
	int obs_dim <- 18;
	int act_dim <- 9;

	// The eight lines that win a game.
	list<list<int>> lines <- [
		[0,1,2], [3,4,5], [6,7,8],
		[0,3,6], [1,4,7], [2,5,8],
		[0,4,8], [2,4,6]
	];

	init {
		do init_petz();
	}

	action init_petz() {
		create PetzAgent {
			agents <- copy(myself.rl_agents);
			possible_agents <- copy(myself.rl_agents);
			observation_spaces <- possible_agents as_map (each::[
				"type"::"Box",
				"low"::list_with(myself.obs_dim, 0.0),
				"high"::list_with(myself.obs_dim, 1.0),
				"shape"::[myself.obs_dim],
				"dtype"::"float"
			]);
			action_spaces <- possible_agents as_map (each::[
				"type"::"Discrete",
				"n"::myself.act_dim
			]);
			loop a over: possible_agents {
				observations[a] <- list_with(myself.obs_dim, 0.0);
				rewards[a] <- 0.0;
				terminations[a] <- false;
				truncations[a] <- false;
				infos << a::[];
			}
			do publish();
		}
		write "AEC tictactoe ready";
	}

	int mark_of (string who) {
		return who = "player_1" ? 1 : 2;
	}

	// Own marks first, then the opponent's — tictactoe_v3's (3,3,2) planes, flattened.
	list<float> observe_player (string who) {
		int me <- mark_of(who);
		int them <- 3 - me;
		list<float> obs <- [];
		loop i from: 0 to: 8 {
			obs << ((board[i] = me) ? 1.0 : 0.0);
		}
		loop i from: 0 to: 8 {
			obs << ((board[i] = them) ? 1.0 : 0.0);
		}
		return obs;
	}

	list<int> legal_moves() {
		list<int> mask <- [];
		loop i from: 0 to: 8 {
			mask << ((board[i] = 0) ? 1 : 0);
		}
		return mask;
	}

	// Applied at once: the mark is on the board before the opponent observes. That is the
	// whole reason a board game needs the AEC API rather than the parallel one.
	action place (string who, int cell) {
		if (game_over) {
			return;
		}
		if (cell >= 0 and cell <= 8 and board[cell] = 0) {
			board[cell] <- mark_of(who);
		}
		do check_end();
	}

	action check_end() {
		loop l over: lines {
			int a <- board[l[0]];
			if (a != 0 and board[l[1]] = a and board[l[2]] = a) {
				winner <- a;
				game_over <- true;
				return;
			}
		}
		if (not (board contains 0)) {
			winner <- 3;			// board full, nobody lined up three
			game_over <- true;
		}
	}

	action reset_board() {
		board <- list_with(9, 0);
		winner <- 0;
		game_over <- false;
	}
}


species PetzAgent {

	list<string> agents;
	list<string> possible_agents;
	map<string, map> observation_spaces;
	map<string, map> action_spaces;

	map<string, list<float>> observations;
	map<string, float> rewards;
	map<string, bool> terminations;
	map<string, bool> truncations;
	map<string, map> infos;
	map data;

	list<float> observe_one (string who) {
		return world.observe_player(who);
	}

	action set_action_one (string who, string action_json) {
		string target <- who;
		int cell <- int(from_json(action_json));
		ask world {
			do place(who: target, cell: cell);
		}
	}

	// Nothing to advance: the move IS the step. Kept because GamaAECEnv closes every round.
	action end_round() {
	}

	action publish() {
		loop a over: possible_agents {
			observations[a] <- world.observe_player(a);
			// +1 to the winner, -1 to the loser, 0 for a draw — tictactoe_v3's convention.
			int me <- world.mark_of(a);
			rewards[a] <- ((world.winner = 0) or (world.winner = 3)) ? 0.0
				: ((world.winner = me) ? 1.0 : -1.0);
			terminations[a] <- world.game_over;
			truncations[a] <- false;
			infos[a] <- ["action_mask":: world.legal_moves()];
		}
		data <- [
			"Observations":: observations,
			"Rewards":: rewards,
			"Terminations":: terminations,
			"Truncations":: truncations,
			"Infos":: infos
		];
	}

	// Python calls this after env.reset(): reloading the experiment is far slower than
	// clearing nine cells, and the rules are deterministic so there is no seed to set.
	action new_game() {
		ask world {
			do reset_board();
		}
		do publish();
	}
}


experiment aec type: gui {
	output {
		display board type: 2d {
			graphics "grid" {
				loop i from: 0 to: 8 {
					point p <- {(i mod 3) * 10 + 5, (i div 3) * 10 + 5};
					draw square(9.5) at: p color: #white border: #black;
					if (board[i] = 1) {
						draw "X" at: p + {-2, 2} color: #crimson font: font("Arial", 24, #bold);
					} else if (board[i] = 2) {
						draw "O" at: p + {-2, 2} color: #royalblue font: font("Arial", 24, #bold);
					}
				}
			}
		}
		monitor "winner" value: winner;
		monitor "game over" value: game_over;
	}
}
