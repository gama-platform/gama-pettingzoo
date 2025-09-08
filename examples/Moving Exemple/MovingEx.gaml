/**
* Name: movingex
* Based on the internal empty template. 
* Author: VezP
* Tags: 
*/


model movingex

global {
	init {
		create mover number: 10;
	}
}

species mover skills: [moving] {
	point dest;
	float speed <- 1.0;
	
	init {
		dest <- rnd({100.0, 100.0});
	}
	
	reflex {
		if location distance_to dest < speed {
			dest <- rnd({100.0, 100.0});
		}
		
		do goto	target: dest speed: speed;
	}
	
	aspect default {
		draw circle(2) color: #red;
	}
}

experiment main {
	reflex {
		ask simulation {
			save (snapshot("Visualisation")) to: "snapshots/frame_" + cycle + ".png" rewrite: true;
		}
	}
	
	output {
		display Visualisation {
			species mover;
		}
	}
}

