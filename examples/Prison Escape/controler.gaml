
model prison_escape_controler

import "PrisonEscape.gaml" as PrisonEscape

global {
	int escape_y;
	int escape_x;
	int guard_y;
	int guard_x;
	int prisoner_y;
	int prisoner_x;
	int timestep;
	list<string> possible_agents <- ["prisoner", "guard"];
	
	cell esc;
	guard grd;
	prisoner prs;
	
	int gs;
	
	init {
		create PrisonEscape.rl_ctrl;
//		write type_of(PrisonEscape.rl_ctrl[0]);
		
		ask PrisonEscape.rl_ctrl[0].simulation {
//			write "world mod: " + self.world;
			esc <- escape;
//			write "esc type: " + type_of(esc);
//			write "esc: " + esc;
//			write "controler escape: " + escape;
			escape_y <- escape.grid_y;
			escape_x <- escape.grid_x;
			grd <- first(guard);
//			write "grd: " + grd;
			prs <- first(prisoner);
//			write "prs: " + prs;
			gs <- grid_size;
		}
		
//		escape_y <- esc.grid_y;
//		escape_x <- esc.grid_x;
		guard_y <- grd.my_cell.grid_y;
		guard_x <- grd.my_cell.grid_x;
		prisoner_y <- prs.my_cell.grid_y;
		prisoner_x <- prs.my_cell.grid_x;
		
		create PetzAgent {
			agents <- copy(myself.possible_agents);
			possible_agents <- copy(myself.possible_agents);
			observation_spaces <- agents as_map (each::["type"::"MultiDiscrete", "nvec"::list_with(3, gs * gs)]);
			action_spaces <- agents as_map (each::["type"::"Discrete", "n"::4]);
		}
		
//		create PetzAgent;
//		PetzAgent[0].agents <- copy(possible_agents);
//		PetzAgent[0].possible_agents <- copy(possible_agents);
//		PetzAgent[0].observation_spaces <- agents as_map (each::["type"::"MultiDiscrete", "nvec"::list_with(3, gs * gs)]);
//		PetzAgent[0].action_spaces <- agents as_map (each::["type"::"Discrete", "n"::4]);
		
		write "observation spaces: " + PetzAgent[0].observation_spaces;
		write "action spaces: " + PetzAgent[0].action_spaces;

		loop a over: PetzAgent[0].agents {
			PetzAgent[0].observations << a::[prisoner_x + gs * prisoner_y, guard_x + gs * guard_y, escape_x + gs * escape_y];

			PetzAgent[0].infos << a::[];
		}
		
		write "observations: " + PetzAgent[0].observations;
		write "infos: " + PetzAgent[0].infos;
		write "agents: " + PetzAgent[0].agents;
		
	}
	
	reflex simulate_step {
		int prisoner_action <- int(PetzAgent[0].actions["prisoner"]);
		int guard_action <- int(PetzAgent[0].actions["guard"]);
		
		ask PrisonEscape.rl_ctrl[0] {
			if prisoner_action = 0 and prisoner_x > 0 {
				prisoner[0].next_cell <- prisoner[0].my_cell.cell_left;
				prisoner_x <- prisoner_x - 1;
			}
			else if prisoner_action = 1 and prisoner_x < (gs - 1) {
				prisoner[0].next_cell <- prisoner[0].my_cell.cell_right;
				prisoner_x <- prisoner_x + 1;
			}
			else if prisoner_action = 2 and prisoner_y > 0 {
				prisoner[0].next_cell <- prisoner[0].my_cell.cell_up;
				prisoner_y <- prisoner_y - 1;
			}
			else if prisoner_action = 3 and prisoner_y < (gs - 1) {
				prisoner[0].next_cell <- prisoner[0].my_cell.cell_down;
				prisoner_y <- prisoner_y + 1;
			}
			else {
				prisoner[0].next_cell <- prisoner[0].my_cell;
			}
			
			
			if guard_action = 0 and guard_x > 0 {
				guard[0].next_cell <- guard[0].my_cell.cell_left;
				guard_x <- guard_x - 1;
			}
			else if guard_action = 1 and guard_x < (gs - 1) {
				guard[0].next_cell <- guard[0].my_cell.cell_right;
				guard_x <- guard_x + 1;
			}
			else if guard_action = 2 and guard_y > 0 {
				guard[0].next_cell <- guard[0].my_cell.cell_up;
				guard_y <- guard_y - 1;
			}
			else if guard_action = 3 and guard_y < (gs - 1) {
				guard[0].next_cell <- guard[0].my_cell.cell_down;
				guard_y <- guard_y + 1;
			}
			else {
				guard[0].next_cell <- guard[0].my_cell;
			}
		}
		
		ask PrisonEscape.rl_ctrl[0].simulation {
			do _step_;
		}
		
		loop a over: PetzAgent[0].agents {
			PetzAgent[0].terminations << a::false;
			PetzAgent[0].rewards << a::0;
			PetzAgent[0].truncations << a::false;
		}
		
		if prisoner_x = guard_x and prisoner_y = guard_y {
			PetzAgent[0].rewards <- ["prisoner"::-1, "guard"::1];
			PetzAgent[0].terminations <- ["prisoner"::true, "guard"::true];
		}
		
		else if prisoner_x = escape_x and prisoner_y = escape_y {
			PetzAgent[0].rewards <- ["prisoner"::1, "guard"::-1];
			PetzAgent[0].terminations <- ["prisoner"::true, "guard"::true];
		}
		
		if timestep > 100 {
			PetzAgent[0].rewards <- ["prisoner"::0, "guard"::0];
			PetzAgent[0].terminations <- ["prisoner"::true, "guard"::true];
		}
		timestep <- timestep + 1;
		
		loop a over: PetzAgent[0].agents {
			PetzAgent[0].observations << a::[prisoner_x + gs * prisoner_y, guard_x + gs * guard_y, escape_x + gs * escape_y];

			PetzAgent[0].infos << a::[];
		}
		
		if any(PetzAgent[0].terminations.values()) or all_match(PetzAgent[0].truncations.values(), each = true) {
			PetzAgent[0].agents <- [];
		}
		
		ask PetzAgent { do update_data; }
		
		write "Observations: " + PetzAgent[0].observations;
		write "Rewards: " + PetzAgent[0].rewards;
		write "Terminations: " + PetzAgent[0].terminations;
		write "Truncations: " + PetzAgent[0].truncations;
	}
}
//------------ Needed to send and receive data to/from Python ------------
species PetzAgent {
	list<string> agents;
	list<string> possible_agents;
	map<string, unknown> observation_spaces;
	map<string, unknown> action_spaces;
	
	map<string, unknown> observations;
	map<string, float> rewards;
	map<string, bool> terminations;
	map<string, bool> truncations;
	map<string, map<string, unknown>> infos;
	
	map<string, unknown> actions;
	
	map<string, map<string, unknown>> data;
	
	action update_data {
		data <- ["Observations"::observations, "Rewards"::rewards, "Terminations"::terminations, "Truncations"::truncations, "Infos"::infos];
	}
}
//---------------------------------------------------------------------

experiment main {
	init {
		write "observation spaces: " + PetzAgent[0].observation_spaces;
		write "action spaces: " + PetzAgent[0].action_spaces;
	}
	
//	reflex snapshot // take a snapshot of the current distribution model instance
//	{
//		write("SNAPPING___________________________________ " + cycle);
//		ask simulation
//		{	
//			save (snapshot("Visualisation")) to: "snapshot/" + cycle + ".png" rewrite: true;		
//			
//		}
//	}

	output {
		display "Visualisation" type: 2d{
			graphics "vis" {
				ask PrisonEscape.rl_ctrl[0].simulation {
					loop c over: cell {
						if c = escape {
							draw c color: #brown border: #grey;
						}
						else {
							draw c color: #white border: #grey;
						}
					}
					draw circle(40/gs) color: #orange at: prisoner[0].my_cell.location;
					draw circle(40/gs) color: #blue at: guard[0].my_cell.location;
					write "prisoner location: " + prisoner[0].my_cell.location;
					write "guard location: " + guard[0].my_cell.location;
					
				}
//				write "prisoner location: " + PrisonEscape.rl_ctrl[0].simulation.prisoner[0].my_cell.location;
//				
//				draw circle(40/gs) color: #orange at: PrisonEscape.rl_ctrl[0].simulation.prisoner[0].my_cell.location;
//				draw circle(40/gs) color: #blue at: PrisonEscape.rl_ctrl[0].simulation.guard[0].my_cell.location;
			}
		}
	}
}