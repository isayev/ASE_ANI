for i in {0..4}
do
	mkdir train$i
        cd train$i
        mkdir networks
        echo "HDAtomNNP-Trainer -i ../inputtrain.ipt -d ../../cache/cache-data-$i/ -p 1.0 -m > output.opt"
        HDAtomNNP-Trainer -i ../inputtrain.ipt -d ../../cache/cache-data-$i/ -p 1.0 -m > output.opt
	sleep 1
        cd ../
done

