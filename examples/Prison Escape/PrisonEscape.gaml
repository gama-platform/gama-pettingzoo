

model prison_escape

global {
	int grid_size <- 7;
	cell escape;
	
	init {
		escape <- first(cell);
		loop while: [first(cell), last(cell)] contains escape { 
			escape <- one_of(cell);
		}
		escape.color <- #brown;
//		write "escape: " + escape;
		
		create guard;
		create prisoner;
	}
	
	point distance_to(cell start, cell end) {
		
		int x_dist <- end.grid_x - start.grid_x;
		int y_dist <- end.grid_y - start.grid_y;
		
		return {x_dist, y_dist};
	}

}

species PettingZooAgent {
	
}

species guard {
	cell my_cell;
	point location;
	
	cell next_cell;
	
	init {
		my_cell <- first(cell);
		location <- my_cell.location;
	}
	
	reflex searching {
		if next_cell = nil {
			point dist <- world.distance_to(my_cell, prisoner[0].my_cell);
			
			if (abs(dist.x) > abs(dist.y)) and (dist.x > 0) {
				next_cell <- my_cell.cell_right;		
			}
			else if (abs(dist.x) > abs(dist.y)) and (dist.x < 0) {
				next_cell <- my_cell.cell_left;
			}
			else if (abs(dist.x) <= abs(dist.y)) and (dist.y > 0) {
				next_cell <- my_cell.cell_down;		
			}
			else if (abs(dist.x) <= abs(dist.y)) and (dist.y < 0) {
				next_cell <- my_cell.cell_up;
			}
			else if dist.x = 0 and dist.y = 0 {
				next_cell <- my_cell;
				ask world {do pause;}
			}
		}
		
		my_cell <- next_cell;
		location <- my_cell.location;
		next_cell <- nil;
	}
	
	aspect default {
		draw circle(40/grid_size) color: #blue;
	}
}

species prisoner {
	cell my_cell;
	point location;
	
	cell next_cell;
	
	init {
		my_cell <- last(cell);
		location <- my_cell.location;
	}
	
	reflex escaping {
		if next_cell = nil {
			point dist <- world.distance_to(my_cell, world.escape);
			
			if (abs(dist.x) >= abs(dist.y)) and (dist.x > 0) {
				next_cell <- my_cell.cell_right;		
			}
			else if (abs(dist.x) >= abs(dist.y)) and (dist.x < 0) {
				next_cell <- my_cell.cell_left;
			}
			else if (abs(dist.x) < abs(dist.y)) and (dist.y > 0) {
				next_cell <- my_cell.cell_down;		
			}
			else if (abs(dist.x) < abs(dist.y)) and (dist.y < 0) {
				next_cell <- my_cell.cell_up;
			}
			else if dist.x = 0 and dist.y = 0 {
				next_cell <- my_cell;
				ask world {do pause;}
			}
		}
		
		my_cell <- next_cell;
		location <- my_cell.location;
		next_cell <- nil;
	}
	
	aspect default {
		draw circle(40/grid_size) color: #orange;
	}
}

grid cell width: grid_size height: grid_size{
	cell cell_up;
	cell cell_down;
	cell cell_right;
	cell cell_left;
	
	init {
		// Find neighboring cells in each direction
		cell_up <- neighbors first_with (each.grid_y < grid_y);
		cell_down <- neighbors first_with (each.grid_y > grid_y);
		
		cell_right <- neighbors first_with (each.grid_x > grid_x);
		cell_left <- neighbors first_with (each.grid_x < grid_x);
	}
}

experiment petz_env {
	
	init {
		
//		bool delete_done <- delete_file("snapshot/*");
		file nf <- new_folder("snapshot/");
		write "nf contents: " + nf.contents;
	}
	
	output {
		display "render" type: 2d{
			grid cell border: #grey;
			species guard;
			species prisoner;
		}
	}
	
	reflex snapshot // take a snapshot of the current distribution model instance
	{
		write("SNAPPING___________________________________ " + cycle);
		ask simulation {
			save (snapshot("render")) to: "snapshot/frame" + cycle + ".png" rewrite: true format: "image";
		}
	}
}

experiment rl_ctrl type: gui {
}








