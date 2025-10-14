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
	int BLINKY <- 4;
	int PINKY <- 5;
	int INKY <- 6;
	int CLYDE <- 7;
	
	int SCATTER <- 0;
	int CHASE <- 1;
	int FREIGHT <- 2;
	int SPAWN <- 3;
	
	float dt <- 0.1;
	
	geometry shape <- rectangle(NCOLS, NROWS);
	
	NodeGroup nodes;
	PelletGroup pellets;
	
	point homekey;
	
	int lives <- 5;
	
	init {
		
		create NodeGroup with: ["level"::"../../../includes/PacMan/maze1.txt"];
		nodes <- first(NodeGroup);
		ask nodes {
			do setPortalPair({0, 17}, {27, 17});
			world.homekey <- createHomeNodes(11.5, 14.0);
			do connectHomeNodes(world.homekey, {12, 14}, LEFT);
			do connectHomeNodes(world.homekey, {15, 14}, RIGHT);
		}
		create PelletGroup with: ["pelletfile"::"../../../includes/PacMan/maze1.txt"];
		pellets <- first(PelletGroup);
		
		create PacMan {
			do setStartNode(nodes.nodesLUT[{15, 26}]);
			do setBetweenNodes(LEFT);
		}
		
		
		create GhostGroup {
//			pacman <- PacMan[0];
			node <- nodes.getStartTempNode();
		}
		
		ask GhostGroup {
			Node spawnNode <- nodes.nodesLUT[{2 + 11.5, 3 + 14}];
			do setSpawnNode(spawnNode);
		}
		
		ask Blinky { do setStartNode(nodes.nodesLUT[{2+11.5, 0+14}]); }
		ask Pinky { do setStartNode(nodes.nodesLUT[{2+11.5, 3+14}]); }
		ask Inky { do setStartNode(nodes.nodesLUT[{0+11.5, 3+14}]); }
		ask Clyde { do setStartNode(nodes.nodesLUT[{4+11.5, 3+14}]); }
	}
	
	reflex checkGhostEvents {
		ask GhostGroup[0].ghosts {
			if PacMan[0].collideGhost(self) {
				if mode.current = FREIGHT {
					pacman.visible <- false;
					visible <- false;
					
					do startSpawn;
				}
				else if mode.current != SPAWN {
					if pacman.alive {
						lives <- lives - 1;
						ask pacman { do died; }
						ask GhostGroup { do hide; }
						if lives <= 0 {
							ask myself { do pause; }
						}
						else {
							ask myself { do resetLevel; }
						}
					}
				}
			}
		}
	}
	
	action resetLevel {
		ask PacMan { do reset; }
		ask GhostGroup { do reset; }
	}
}

species Entity {
	int name_id;
	Node cur_node;
	Node startNode;
	Node spawnNode;
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
	
	action setBetweenNodes(int direc) {
		if cur_node.neighbors[direc] != nil {
			target <- cur_node.neighbors[direc];
			location <- (cur_node.location + target.location) / 2.0;
		}
	}
	
	action setStartNode(Node stNode) {
		cur_node <- stNode;
		startNode <- stNode;
		target <- stNode;
		do setPosition();
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
	
	action normalMode {
		speed <- 1.0;
	}
	
	action reset {
		do setStartNode(startNode);
		direction <- NOOP;
		speed <- 1.0;
		visible <- true;
	}
	
	aspect default {
		if visible {
			draw circle(radius) color: color;
		}
	}
}

species PacMan parent: Entity{
	cell cur_cell;
	bool alive <- true;
	
	init {
		name_id <- PACMAN;
		color <- #yellow;
		direction <- LEFT;
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
			if pellet.name_id = POWERPELLET {
				ask GhostGroup[0].ghosts {
					do startFreight;
				}
			}
		}
	}
	
	
	
	Pellet eatPellets(list<Pellet> pelletList) {
		loop pellet over: pelletList {
			if collideCheck(pellet, pellet.collideRadius) {
				return pellet;
			}
		}
		return nil;
	}
	
	bool collideGhost(Ghost ghost) {
		return collideCheck(ghost, ghost.radius);
	}
	
	bool collideCheck(agent other, float collideRadius) {
		let d <- location - other.location;
		let dsqr <- d.x^2 + d.y^2;
		let rsqr <- (radius + collideRadius)^2;
		if dsqr <= rsqr {
			return true;
		}
		return false;
	}
	
	action reset {
		invoke reset;
		direction <- LEFT;
		do setBetweenNodes(LEFT);
		alive <- true;		
	}
	
	action died {
		alive <- false;
		direction <- NOOP;
	}
}

species Ghost parent: Entity{
	int points;
	PacMan pacman;
	ModeController mode;
	Ghost blinky;
	Node homeNode;
	
	init {
		name_id <- GHOST;
		points <- 200;
		goal <- {0, 0};
		
		create ModeController returns: modes {
			entity <- myself;
		}
		mode <- first(modes);
	}
	
	reflex preupdate {
		ask mode {
			do update;
		}
		if mode.current = SCATTER {
			do scatter;
		}
		else if mode.current = CHASE {
			do chase;
		}
	}
	
	action scatter {
//		color <- #white;
		goal <- {0, 0};
	}
	
	action chase {
//		color <- #red;
		goal <- pacman.location;
	}
	
	action directionMethod(list<int> possible_directions) {
		if mode.current = FREIGHT {
			return randomDirection(possible_directions);
		}
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
	
	action startFreight {
		ask mode {
			do setFreightMode;
		}
		if mode.current = FREIGHT {
			speed <- 0.5;
		}
	}
	
	action spawn {
//		color <- #blue;
		goal <- spawnNode.location;
	}
	
	action setSpawnNode(Node node) {
		spawnNode <- node;
	}
	
	action startSpawn {
		ask mode {
			do setSpawnMode;
		}
		if mode.current = SPAWN {
			speed <- 1.5;
			do spawn;
		}
	}
	
	action reset {
		invoke reset;
		points <- 200;
	}
}

species Blinky parent: Ghost{
	init {
		name_id <- BLINKY;
		color <- #red;
	}
}

species Pinky parent:Ghost {
	init {
		name_id <- PINKY;
		color <- #pink;
	}
	
	action scatter {
		goal <- {NCOLS, 0};
	}
	
	action chase {
		goal <- pacman.location + pacman.directions[pacman.direction] * 4;
	}
}

species Inky parent: Ghost {
	init {
		name_id <- INKY;
		color <- #teal;
	}
	
	action scatter {
		goal <- {NCOLS, NROWS};
	}
	
	action chase {
		let vec1 <- pacman.location + pacman.directions[pacman.direction] * 2;
		let vec2 <- (vec1 - blinky.location) * 2;
		goal <- blinky.location + vec2;
	}
}

species Clyde parent: Ghost {
	init {
		name_id <- CLYDE;
		color <- #orange;
	}
	
	action scatter {
		goal <- {0, NROWS};
	}
	
	action chase {
		let d <- pacman.location - location;
		let ds <- d.x^2 + d.y^2;
		if ds <= 8^2 {
			do scatter;
		}
		else {
			goal <- pacman.location + pacman.directions[pacman.direction] * 4;
		}
	}
}

species GhostGroup {
	Node node;
	PacMan pacman;
	Ghost blinky;
	Ghost pinky;
	Ghost inky;
	Ghost clyde;
	
	list<Ghost> ghosts;
	
	init {
		pacman <- PacMan[0];
		create Blinky {
			spawnNode <- myself.node;
			self.pacman <- myself.pacman;
		}
		blinky <- Blinky[0];
		
		create Pinky {
			spawnNode <- myself.node;
			self.pacman <- myself.pacman;
		}
		pinky <- Pinky[0];
		
		create Inky {
			spawnNode <- myself.node;
			self.pacman <- myself.pacman;
			self.blinky <- myself.blinky;
		}
		inky <- Inky[0];
		
		create Clyde {
			spawnNode <- myself.node;
			self.pacman <- myself.pacman;
		}
		clyde <- Clyde[0];
		
		ghosts <- [blinky, pinky, inky, clyde];
	}
	
	action startFreight {
		ask ghosts {
			do startFreight;
		}
		do resetPoints;
	}
	
	action setSpawnNode(Node sNode) {
		ask ghosts {
			do setSpawnNode(sNode);
		}
	}
	
	action updatePoints {
		ask ghosts {
			points <- points * 2;
		}
	}
	
	action resetPoints {
		ask ghosts {
			points <- 200;
		}
	}
	
	action reset {
		ask Ghost {
			do reset;
		}
	}
	
	action hide {
		ask ghosts {
			visible <- false;
		}
	}
	
	action show {
		ask ghosts {
			visible <- true;
		}
	}
}

species MainMode {
	float timer <- 0.0;
	int mode <- SCATTER;
	int time <- 7;
	
	action update {
		timer <- timer + dt;
		
		if timer >= time {
			if mode = SCATTER {
				do chase;
			}
			else if mode = CHASE {
				do scatter;
			}
		}
	}
	
	action chase {
		mode <- CHASE;
		time <- 20;
		timer <- 0.0;
	}
	
	action scatter {
		mode <- SCATTER;
		time <- 7;
		timer <- 0.0;
	}
}

species ModeController {
	float timer <- 0.0;
	int time;
	MainMode mainmode;
	int current;
	Entity entity;
	
	init {
		create MainMode returns: mainmodes;
		mainmode <- first(mainmodes);
		current <- mainmode.mode;
	}
	
	action update {
		ask mainmode {
			do update;
		}
		if current = FREIGHT {
			timer <- timer + dt;
			if timer >= time {
				time <- nil;
				ask entity {
					do normalMode;
				}
				current <- mainmode.mode;
			}
		}
		else if current in [SCATTER, CHASE] {
			current <- mainmode.mode;
		}
		
		if current = SPAWN {
			if entity.cur_node = entity.spawnNode {
				ask entity {
					do normalMode;
				}
				current <- mainmode.mode;
			}
		}
	}
	
	action setSpawnMode {
		if current = FREIGHT {
			current <- SPAWN;
		}
	}
	
	action setFreightMode {
		if current in [SCATTER, CHASE] {
			timer <- 0.0;
			time <- 7;
			current <- FREIGHT;
		}
		else if current = FREIGHT {
			timer <- 0.0;
		}
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
	point homekey;
	
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
	
	action createNodeTable(matrix data, float xoffset<-0, float yoffset<-0) {
		loop row over: range(int(data.dimension.x) - 1) {
			loop col over: range(int(data.dimension.y) - 1) {
				if data[{row, col}] in nodeSymbols {
					create Node with: ["location"::{col + xoffset, row + yoffset}] returns: new_node;
					nodesLUT[{col + xoffset, row + yoffset}] <- new_node[0];
				}
			}
		}
	}
	
	action connectHorizontally(matrix data, float xoffset<-0, float yoffset<-0) {
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
	
	action connectVertically(matrix data, float xoffset<-0, float yoffset<-0) {
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
	
	Node getNodeFromPixels(float xpixel, float ypixel) {
		if {xpixel, ypixel} in nodesLUT.keys {
			return nodesLUT[{xpixel, ypixel}];
		}
		return nil;
	}
	
	Node getNodeFromTiles(float col, float row) {
		return getNodeFromPixels(col, row);
	}
	
	Node getStartTempNode {
		list<Node> list_nodes <- list(nodesLUT.values);
		return list_nodes[0];
	}
	
	point createHomeNodes(float xoffset, float yoffset) {
		matrix homedata <- matrix([
			['X','X','+','X','X'],
			['X','X','.','X','X'],
         	['+','X','.','X','+'],
         	['+','.','+','.','+'],
         	['+','X','X','X','+']]);
		do createNodeTable(homedata, xoffset, yoffset);
		do connectHorizontally(homedata, xoffset, yoffset);
		do connectVertically(homedata, xoffset, yoffset);
		homekey <- {xoffset + 2, yoffset};
		return homekey;
	}
	
	action connectHomeNodes(point hk, point otherkey, int direction) {
		nodesLUT[hk].neighbors[direction] <- nodesLUT[otherkey];
		nodesLUT[otherkey].neighbors[direction * -1] <- nodesLUT[hk];
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
			species Blinky;
			species Pinky;
			species Inky;
			species Clyde;
			
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