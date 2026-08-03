/**
* Name: Wildfire
* Description: Turn-based (AEC) example for gama-pettingzoo — a team of firefighters
*              containing a spreading wildfire on a grid.
*
* Why this needs the AEC environment rather than the parallel one:
* a firefighter puts out the fire on the cell it moves onto, and that takes effect AT ONCE.
* The next firefighter of the same round therefore sees a map where that flame is already
* gone, and goes somewhere else. Under the parallel API all of them would decide against the
* same start-of-round map, converge on the nearest flame, and waste their turn — the very
* coordination problem the model is meant to pose would be unlearnable.
*
* The fire spreads in a GAMA cycle rather than inside end_round(), so the display refreshes
* between rounds. Drive it from Python with steps_per_round=1.
*
* GAML-side contract expected by GamaAECEnv:
*   PetzAgent[0].observe_one("<agent>") / set_action_one("<agent>", '<json>')
*   PetzAgent[0].end_round() / publish() / data
*/

model Wildfire

global {

	int grid_size <- 20;
	int nb_firefighters <- 3;
	// Chance that a burning cell sets a given vegetated neighbour alight, per round.
	//
	// These two are what make the task winnable. A cell tries to ignite each of its ~8
	// vegetated neighbours once per round for `burn_duration` rounds, so it spawns roughly
	// 8 * burn_duration * spread_probability new fires over its life. Keep that product near
	// 1: below it the fire dies unaided and the agents are decorative, well above it three
	// firefighters dousing one cell each per round can never catch up and the task is not
	// hard but impossible. At 0.06 / 2 the fire creeps and a coordinated team contains it.
	float spread_probability <- 0.06;
	int initial_fires <- 3;
	int burn_duration <- 2;			// rounds a cell stays alight before turning to ash
	int max_rounds <- 60;

	int round_index <- 0;
	int decided_this_round <- 0;
	bool round_closed <- false;

	// Set by the spread reflex so publish() can charge everyone for the round's damage.
	int burnt_this_round <- 0;

	int obs_dim <- 22;
	int act_dim <- 5;			// stay, north, south, east, west

	list<string> rl_agents;
	map<string, Firefighter> firefighter_by_name;

	init {
		ask initial_fires among cell {
			state <- 2;
		}
		// Deploy the team within reach of the front. Dropped at random on a 20x20 grid they
		// spent most of the episode walking, and the fire was decided before they arrived.
		create Firefighter number: nb_firefighters {
			cell seat <- one_of(cell where (each.state = 2));
			list<cell> nearby <- (cell where (each.state != 2))
				where ((each.location distance_to seat.location) < 6.0);
			my_cell <- empty(nearby) ? one_of(cell where (each.state != 2)) : one_of(nearby);
			location <- my_cell.location;
		}
		do init_petz();
	}

	action init_petz() {
		rl_agents <- Firefighter collect (string(each));
		loop f over: Firefighter {
			firefighter_by_name[string(f)] <- f;
		}
		create PetzAgent {
			agents <- copy(myself.rl_agents);
			possible_agents <- copy(myself.rl_agents);
			// Every observation component is a flag or a ratio -> bounds [0, 1].
			observation_spaces <- possible_agents as_map (each::[
				"type"::"Box",
				"low"::list_with(myself.obs_dim, 0.0),
				"high"::list_with(myself.obs_dim, 1.0),
				"shape"::[myself.obs_dim],
				"dtype"::"float"
			]);
			// One move per turn; the firefighter extinguishes the cell it lands on.
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
		write "AEC wildfire ready: " + length(rl_agents) + " firefighters, "
			+ length(cell where (each.state = 2)) + " fires";
	}

	// 8 burning flags + 8 vegetation flags around the agent, then 4 global signals.
	list<float> observe_firefighter (string who) {
		Firefighter f <- firefighter_by_name[who];
		list<cell> around <- f.my_cell.neighbors;
		list<float> obs <- [];
		loop i from: 0 to: 7 {
			cell c <- i < length(around) ? around[i] : nil;
			obs << ((c != nil) and (c.state = 2)) ? 1.0 : 0.0;
		}
		loop i from: 0 to: 7 {
			cell c <- i < length(around) ? around[i] : nil;
			obs << ((c != nil) and (c.state = 1)) ? 1.0 : 0.0;
		}
		obs << f.my_cell.grid_x / float(grid_size);
		obs << f.my_cell.grid_y / float(grid_size);
		obs << length(cell where (each.state = 2)) / float(grid_size * grid_size);
		// Turn position within the round: 0 for the first to play, (n-1)/n for the last.
		obs << float(decided_this_round) / float(length(rl_agents));
		// Bearing to the nearest flame, centred on 0.5. Without it an agent has no gradient
		// toward a fire it cannot already touch, and finding one by random walk on a 20x20
		// grid is hopeless -- the task would be unlearnable rather than merely hard.
		list<cell> fires <- cell where (each.state = 2);
		cell target <- empty(fires) ? nil : (fires closest_to f.my_cell);
		obs << (target = nil) ? 0.5 : (0.5 + 0.5 * (target.grid_x - f.my_cell.grid_x) / float(grid_size));
		obs << (target = nil) ? 0.5 : (0.5 + 0.5 * (target.grid_y - f.my_cell.grid_y) / float(grid_size));
		return obs;
	}

	// Applied at once, not queued: the flame is out NOW, so the next firefighter of the
	// round sees a different map. This is the crux of the AEC formulation.
	action move_and_douse (string who, int move) {
		Firefighter f <- firefighter_by_name[who];
		int nx <- f.my_cell.grid_x + (move = 3 ? 1 : (move = 4 ? -1 : 0));
		int ny <- f.my_cell.grid_y + (move = 1 ? -1 : (move = 2 ? 1 : 0));
		nx <- max(0, min(grid_size - 1, nx));
		ny <- max(0, min(grid_size - 1, ny));

		f.my_cell <- cell[nx, ny];
		f.location <- f.my_cell.location;
		f.doused_this_round <- 0;
		if (f.my_cell.state = 2) {
			f.my_cell.state <- 1;			// extinguished, vegetation survives
			f.doused_this_round <- 1;
			f.total_doused <- f.total_doused + 1;
		}
		decided_this_round <- decided_this_round + 1;
	}

	action close_round() {
		round_closed <- true;
	}

	// Runs in a GAMA cycle, which is what refreshes the display between rounds.
	reflex spread when: round_closed {
		list<cell> burning <- cell where (each.state = 2);
		list<cell> catching <- [];
		loop b over: burning {
			loop n over: (b.neighbors where (each.state = 1)) {
				if (flip(spread_probability)) {
					catching << n;
				}
			}
		}
		catching <- remove_duplicates(catching);
		ask catching {
			state <- 2;
			burn_time <- 0;
		}
		// Cells burn for several rounds before turning to ash, which is what gives the fire
		// time to grow and the team time to reach it.
		list<cell> spent <- [];
		ask burning {
			burn_time <- burn_time + 1;
			if (burn_time >= burn_duration) {
				state <- 0;
				spent << self;
			}
		}
		burnt_this_round <- length(spent);

		round_index <- round_index + 1;
		decided_this_round <- 0;
		round_closed <- false;
	}
}


grid cell width: 20 height: 20 neighbors: 8 {
	// 0 = burnt, 1 = vegetation, 2 = burning
	int state <- 1;
	// Rounds this cell has been alight. A cell that turned to ash after a single round left
	// the fire no time to spread, and every episode ended itself before the team arrived.
	int burn_time <- 0;
	rgb color -> state = 2 ? #orangered : (state = 1 ? #darkgreen : #dimgray);
}


species Firefighter {
	cell my_cell;
	int doused_this_round <- 0;
	int total_doused <- 0;

	aspect default {
		draw circle(0.45) color: #deepskyblue border: #white;
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
		return world.observe_firefighter(who);
	}

	action set_action_one (string who, string action_json) {
		string target <- who;
		int chosen <- int(from_json(action_json));
		ask world {
			do move_and_douse(who: target, move: chosen);
		}
	}

	action end_round() {
		ask world {
			do close_round();
		}
	}

	action publish() {
		int still_burning <- length(cell where (each.state = 2));
		bool contained <- still_burning = 0;
		loop a over: possible_agents {
			Firefighter f <- world.firefighter_by_name[a];
			observations[a] <- world.observe_firefighter(a);
			// Credit for the flame you personally put out, a shared bill for what burned
			// anyway: individually useful moves are rewarded, but nobody wins if the fire
			// runs away while everyone optimises their own tile.
			rewards[a] <- float(f.doused_this_round)
				- 0.1 * float(world.burnt_this_round)
				+ (contained ? 10.0 : 0.0);
			terminations[a] <- contained;
			truncations[a] <- world.round_index >= world.max_rounds;
		}
		data <- [
			"Observations":: observations,
			"Rewards":: rewards,
			"Terminations":: terminations,
			"Truncations":: truncations,
			"Infos":: infos
		];
	}
}


experiment aec type: gui {
	parameter "Firefighters" var: nb_firefighters init: 3;
	parameter "Spread probability" var: spread_probability init: 0.06 min: 0.0 max: 1.0;
	parameter "Initial fires" var: initial_fires init: 3;

	output {
		display map type: 2d {
			grid cell border: rgb(40, 40, 40);
			species Firefighter aspect: default;
		}
		monitor "round" value: round_index;
		monitor "burning cells" value: length(cell where (each.state = 2));
		monitor "burnt cells" value: length(cell where (each.state = 0));
	}
}
