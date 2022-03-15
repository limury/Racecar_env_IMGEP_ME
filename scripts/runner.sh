#!/bin/bash

cd ../src

#python3 run_map_elites.py $1 100000 160 5000 curiosity curiosity_func               

python3 run_map_elites.py $1 100000 160 5000 nn sbx                            
python3 run_map_elites.py $1 100000 160 5000 nn polynomial                     
python3 run_map_elites.py $1 100000 160 5000 nn polynomial+sbx                 
python3 run_map_elites.py $1 100000 160 5000 nn iso_dd+polynomial+sbx          
#python3 run_map_elites.py $1 100000 160 5000 nn iso_dd                        	
#python3 run_map_elites.py $1 100000 160 5000 nn iso_dd+polynomial              
#python3 run_map_elites.py $1 100000 160 5000 nn iso_dd+sbx                     

#python3 run_map_elites.py $1 100000 160 5000 curiosity iso_dd                        	
#python3 run_map_elites.py $1 100000 160 5000 curiosity polynomial                     
#python3 run_map_elites.py $1 100000 160 5000 curiosity sbx                            
#python3 run_map_elites.py $1 100000 160 5000 curiosity polynomial+sbx                 
#python3 run_map_elites.py $1 100000 160 5000 curiosity iso_dd+polynomial              
#python3 run_map_elites.py $1 100000 160 5000 curiosity iso_dd+sbx                     
#python3 run_map_elites.py $1 100000 160 5000 curiosityso_dd+polynomial+sbx          

#python3 run_map_elites.py $1 100000 160 5000 uniform iso_dd                        	
#python3 run_map_elites.py $1 100000 160 5000 uniform polynomial                     
#python3 run_map_elites.py $1 100000 160 5000 uniform sbx                            
#python3 run_map_elites.py $1 100000 160 5000 uniform polynomial+sbx                 
#python3 run_map_elites.py $1 100000 160 5000 uniform iso_dd+polynomial              
#python3 run_map_elites.py $1 100000 160 5000 uniform iso_dd+sbx                     
#python3 run_map_elites.py $1 100000 160 5000 uniform iso_dd+polynomial+sbx          

#python3 run_map_elites.py $1 100000 160 5000 curiosity-child iso_dd                   
#python3 run_map_elites.py $1 100000 160 5000 curiosity-child polynomial               
#python3 run_map_elites.py $1 100000 160 5000 curiosity-child sbx                      
#python3 run_map_elites.py $1 100000 160 5000 curiosity-child polynomial+sbx           
#python3 run_map_elites.py $1 100000 160 5000 curiosity-child iso_dd+polynomial        
#python3 run_map_elites.py $1 100000 160 5000 curiosity-child iso_dd+sbx               
#python3 run_map_elites.py $1 100000 160 5000 curiosity-child iso_dd+polynomial+sbx    