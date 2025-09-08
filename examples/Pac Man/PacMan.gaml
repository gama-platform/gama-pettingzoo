model pacman

global {
	
	int NROWS <- 36;
	int NCOLS <- 28;
	
	int NOOP <- 0;
	int UP <- 1;
	int RIGHT <- 2;
	int LEFT <- -2;
	int DOWN <- -1;
	int PORTAL <- 3;
	
	int PACMAN <- 0;
	int PELLET <- 1;
	int POWERPELLET <- 2;
	int GHOST <- 3;
	
	float dt <- 0.2;
	
	geometry shape <- rectangle(NCOLS, NROWS);
	
	NodeGroup nodes;
	PelletGroup pellets;
	
	init {
		
		create NodeGroup with: ["level"::"../../../includes/PacMan/maze1.txt"];
		nodes <- first(NodeGroup);
		ask nodes {
			do setPortalPair({0, 17}, {27, 17});
		}
		create PelletGroup with: ["pelletfile"::"../../../includes/PacMan/maze1.txt"];
		pellets <- first(PelletGroup);
		
		create PacMan;
		create Ghost;
	}
}

species Entity {
	int name_id;
	Node cur_node;
	Node target;
	point goal;
	map<int, point> directions <- [NOOP::{0, 0}, UP::{0, -1}, RIGHT::{1, 0}, LEFT::{-1, 0}, DOWN::{0, 1}];
	int direction <- NOOP;
	int new_direction <- NOOP;
	
	action directionMethod(list<int> possible_directions);
	
	float radius <- 0.5;
	float speed <- 1.0;
	rgb color <- #white;
	bool visible <- true;
	bool disablePortal <- false;
	
	init {
		
	}
	
	reflex update {
		location <- location + directions[direction] * speed * dt;
		if overshotTarget() {
			cur_node <- target;
			list<int> possible_directions <- validDirections();
			new_direction <- int(directionMethod(possible_directions));
			if !disablePortal and cur_node.neighbors[PORTAL] != nil{
				cur_node <- cur_node.neighbors[PORTAL];
			}
			target <- getNewTarget(new_direction);
			if target != cur_node {
				direction <- new_direction;
			} else {
				target <- getNewTarget(direction);
			}
			do setPosition();
		}
	}
	
	action setPosition {
		location <- cur_node.location;
	}
	
	bool validDirection(int direc) {
		if direc != NOOP {
			if cur_node.neighbors[direc] != nil {
				return true;
			}
		}
		return false;
	}
	
	list<int> validDirections {
		list<int> possible_directions <- directions.keys where(validDirection(each) and each != (direction * -1));
		if length(possible_directions) = 0 {
			possible_directions << direction * -1;
		}
		return possible_directions;
	}
	
	int randomDirection(list<int> possible_directions) {
		return one_of(possible_directions);
	}
	
	Node getNewTarget(int direc) {
		if validDirection(direc) {
			return cur_node.neighbors[direc];
		}
		return cur_node;
	}
	
	bool overshotTarget {
		if target != nil {
			let vec1 <- target.location - cur_node.location;
			let vec2 <- location - cur_node.location;
			let nodeToTarget <- vec1.x^2 + vec1.y^2;
			let nodeToSelf <- vec2.x^2 + vec2.y^2;
			return nodeToSelf >= nodeToTarget;
		}
		return false;
	}
	
	action reverseDirection {
		direction <- direction * -1;
		let temp <- cur_node;
		cur_node <- target;
		target <- temp;
	}
	
	bool oppositeDirection(int direc) {
		if direc != NOOP {
			if direc = direction * -1 {
				return true;
			}
		}
		return false;
	}
	
	aspect default {
		if visible {
			draw circle(radius) color: color;
		}
	}
}

species PacMan parent: Entity{
	cell cur_cell;
	
	init {
		name_id <- PACMAN;
		color <- #yellow;
		
		cur_node <- nodes.getStartTempNode();
		do setPosition;
		target <- cur_node;
		
	}
	
	reflex update {
		location <- location + directions[direction] * speed * dt;
		if overshotTarget() {
			cur_node <- target;
			if cur_node.neighbors[PORTAL] != nil {
				cur_node <- cur_node.neighbors[PORTAL];
			}
			target <- getNewTarget(new_direction);
			if target != cur_node {
				direction <- new_direction;
			} else {
				target <- getNewTarget(direction);
			}
			
			if target = cur_node {
				direction <- NOOP;
			}
			do setPosition;
//			write "target: " + target + " node: " + cur_node + " direction: " + direction;
		} else {
			if oppositeDirection(new_direction) {
				do reverseDirection;
			}
		}
		
		Pellet pellet <- eatPellets(pellets.pellets);
		if pellet != nil {
			pellets.numEaten <- pellets.numEaten + 1;
			pellets.pellets >> pellet;
		}
	}
	
	
	
	Pellet eatPellets(list<Pellet> pelletList) {
		loop pellet over: pelletList {
			let d <- location - pellet.location;
			let dsqr <- d.x^2 + d.y^2;
			let rsqr <- (pellet.radius + radius)^2;
			if dsqr <= rsqr {
				return pellet;
			}
		}
		return nil;
	}
}

species Ghost parent: Entity{
	int points;
	
	
	init {
		name_id <- GHOST;
		points <- 200;
		goal <- {0, 0};
		
		cur_node <- nodes.getStartTempNode();
		do setPosition;
		target <- cur_node;
	}
	
	action directionMethod(list<int> possible_directions) {
		return goalDirection(possible_directions);
	}
	
	int goalDirection(list<int> possible_directions) {
		list<float> distances <- [];
		loop d over: possible_directions {
			point vec <- cur_node.location + directions[d] - goal;
			let vecsqr <- vec.x^2 + vec.y^2;
			distances << vecsqr;
		}
		int indx <- distances index_of(min(distances));
		return possible_directions[indx];
	}
}

species Pellet {
	int name_id <- PELLET;
	rgb color <- #white;
	float radius <- 0.1;
	float collideRadius <- 0.1;
	int points <- 10;
	bool visible <- true;
	
	aspect default {
		if visible {
			draw circle(radius) color: color;
		}
	}
}

species PowerPellet parent: Pellet{
	int name_id <- POWERPELLET;
	float radius <- 0.3;
	int points <- 50;
	float flashTime <- 2.0;
	float timer <- 0.0;
	
	reflex update {
		timer <- timer + dt;
		if timer >= flashTime {
			visible <- !visible;
			timer <- 0.0;
		}
	}
}

species PelletGroup {
	string pelletfile;
	list<Pellet> pellets <- [];
	list<PowerPellet> powerpellets <- [];
	int numEaten <- 0;
	
	init {
		do createPelletList(pelletfile);
	}
	
	action createPelletList(string filename) {
		matrix data <- readPelletfile(filename);
		loop row over: range(int(data.dimension.x - 1)) {
			loop col over: range(int(data.dimension.y - 1)) {
				if data[{row, col}] in ['.', '+'] {
					create Pellet with: ["location"::{col, row}] returns: new_pellet;
					pellets << first(new_pellet);
				} else if data[{row, col}] in ['P', 'p'] {
					create PowerPellet with: ["location"::{col, row}] returns: new_pp;
					pellets << first(new_pp);
					powerpellets << first(new_pp);
				}
			}
		}
	}
	
	matrix readPelletfile(string filename) {
		matrix pellet_matrix;
		file pellet_file <- text_file(filename);
		pellet_matrix <- matrix(pellet_file collect (string(each) split_with " "));
		return pellet_matrix;
	}
	
	bool isEmpty {
		if length(pellets) = 0 {
			return true;
		}
		return false;
	}
	
	aspect default {
		loop pellet over: pellets {
			if pellet.visible {
				draw circle(pellet.radius) at: pellet.location color: pellet.color;
			}
		}
	}
}

species Node parent: graph_node edge_species: base_edge {
	map<int, Node> neighbors <- [UP::nil, RIGHT::nil, LEFT::nil, DOWN::nil, PORTAL::nil];
	
	bool related_to(Node other) {
		return (other in neighbors.values) ? true : false;
	}
	
	aspect default {
		draw circle(0.2) at: location color: #red;
		loop n over: neighbors.keys {
			if neighbors[n] != nil and n != PORTAL{
				let start_point <- location;
				let end_point <- neighbors[n].location;
				draw line([start_point, end_point]) color: #white;
			}
		}
	}
}

species NodeGroup {
	string level;
	map<point, Node> nodesLUT <- [];
	list<string> nodeSymbols <- ['+', 'P', 'n'];
	list<string> pathSymbols <- ['.', '-', '|', 'p'];
	
	init {
		let data <- readMazeFile(level);
		do createNodeTable(data);
		do connectHorizontally(data);
		do connectVertically(data);
	}
	
	matrix readMazeFile(string filename) {
		matrix maze_mtx;
		file maze_file <- text_file(filename);
		maze_mtx <- matrix(maze_file collect (string(each) split_with " "));
		return maze_mtx;
	}
	
	action createNodeTable(matrix data, int xoffset<-0, int yoffset<-0) {
		loop row over: range(int(data.dimension.x) - 1) {
			loop col over: range(int(data.dimension.y) - 1) {
				if data[{row, col}] in nodeSymbols {
					create Node with: ["location"::{col + xoffset, row + yoffset}] returns: new_node;
					nodesLUT[{col + xoffset, row + yoffset}] <- new_node[0];
				}
			}
		}
	}
	
	action connectHorizontally(matrix data, int xoffset<-0, int yoffset<-0) {
		loop row over: range(int(data.dimension.x) - 1) {
			point key <- nil;
			loop col over: range(int(data.dimension.y) - 1) {
				if data[{row, col}] in nodeSymbols {
					if key = nil {
						key <- {col + xoffset, row + yoffset};
					} else {
						point otherkey <- {col + xoffset, row + yoffset};
						nodesLUT[key].neighbors[RIGHT] <- nodesLUT[otherkey];
						nodesLUT[otherkey].neighbors[LEFT] <- nodesLUT[key];
						key <- otherkey;
					}
				} else if !(data[{row, col}] in pathSymbols){
					key <- nil;				
				}
			}
		}
	}
	
	action connectVertically(matrix data, int xoffset<-0, int yoffset<-0) {
		matrix dataT <- transpose(data);
		loop col over: range(int(dataT.dimension.x) - 1) {
			point key <- nil;
			loop row over: range(int(dataT.dimension.y) - 1) {
				if dataT[{col, row}] in nodeSymbols {
					if key = nil {
						key <- {col + xoffset, row + yoffset};
					} else {
						point otherkey <- {col + xoffset, row + yoffset};
						nodesLUT[key].neighbors[DOWN] <- nodesLUT[otherkey];
						nodesLUT[otherkey].neighbors[UP] <- nodesLUT[key];
						key <- otherkey;
					}
				} else if !(dataT[{col, row}] in pathSymbols) {
					key <- nil;
				}
			}
		}
	}
	
	action setPortalPair(point pair1, point pair2) {
		if pair1 in nodesLUT.keys and pair2 in nodesLUT.keys {
			nodesLUT[pair1].neighbors[PORTAL] <- nodesLUT[pair2];
			nodesLUT[pair2].neighbors[PORTAL] <- nodesLUT[pair1];
		}
	}
	
	Node getStartTempNode {
		list<Node> list_nodes <- list(nodesLUT.values);
		return list_nodes[0];
	}
}

grid cell width: NCOLS height: NROWS{
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
		
		color <- #black;
	}
}

experiment sim {
	output {
		display visualisation type: 2d background: #black{
			grid cell border: #purple wireframe: true;
			species Node;
			species PelletGroup;
			species PacMan;
			species Ghost;
		}
	}
}

experiment play autorun: false{
	float minimum_cycle_duration<-20#ms;
	output {
		display visualisation type: 2d background: #black{
			grid cell border: #purple wireframe: true;
			species Node;
			species PelletGroup;
			species PacMan;
			species Ghost;
			
			event #arrow_up {
				ask PacMan {
					new_direction <- UP;
				}
			}
			event #arrow_right {
				ask PacMan {
					new_direction <- RIGHT;
				}
			}
			event #arrow_left {
				ask PacMan {
					new_direction <- LEFT;
				}
			}
			event #arrow_down {
				ask PacMan {
					new_direction <- DOWN;
				}
			}
			
			event " " {
				ask simulation {
					do resume;
				}
			}
		}
	}
}