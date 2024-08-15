using Tensorflow; 
using Tensorflow.NumPy; 
using static Tensorflow.KerasApi; 
using Tensorflow.Keras.Engine; 
using Tensorflow.Keras.Optimizers; 
using static Tensorflow.Binding; 

public class AlzheimerMRIClassifier
{
    // Dataset'in bulunduğu dosya yolu
    private static string datasetPath = "C:/Users/accag/OneDrive/Masaüstü/Dataset";
    private static int imageSize = 128; // Resimlerin boyutu
    private static int batchSize = 32; // Eğitim sırasında kullanılacak batch boyutu
    private static int epochs = 20; // Eğitim sırasında kaç epoch yapılacağı

    public static void Main()
    {
        CheckDirectoryContents(); // Dataset dizinindeki içeriği kontrol et

        var (trainData, valData, testData) = LoadData(); // Veriyi yükle

        var model = BuildModel(); // Modeli oluştur
        var classWeights = ComputeClassWeights(trainData); // Sınıf ağırlıklarını hesapla

        var classWeightsDict = classWeights as Dictionary<int, float> ?? new Dictionary<int, float>(classWeights);

        var callbacks = new List<ICallback>
        {
            // Özel bir callback eklemek istersen burada tanımlayabilirsin
        };

        // Modeli eğit
        model.fit(trainData, epochs: epochs, validation_data: valData, class_weight: classWeightsDict, callbacks: callbacks);
        EvaluateModel(model, testData); // Modeli değerlendir
    }

    private static (IDatasetV2, IDatasetV2, IDatasetV2) LoadData()
    {
        // Eğitim, doğrulama ve test verilerini yükle
        var trainData = CreateDataset("train");
        var valData = CreateDataset("val");
        var testData = CreateDataset("test");

        return (trainData, valData, testData);
    }

    private static IDatasetV2 CreateDataset(string subset)
    {
        // Sınıf etiketlerini tanımla
        var categories = new[] { "Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented" };

        // Resim dosyalarını al
        var imageFiles = categories.SelectMany(category =>
            Directory.GetFiles(Path.Combine(datasetPath, category), "*.jpg", SearchOption.TopDirectoryOnly)).ToList();

        // Dataset'i karıştır
        var shuffledImageFiles = imageFiles.OrderBy(x => Guid.NewGuid()).ToList();

        // Dosya adlarına göre etiketleri ata
        var labels = shuffledImageFiles.Select(file => categories.ToList().IndexOf(
            categories.First(cat => file.Contains(cat)))).ToArray();

        // TensorFlow Dataset oluştur
        var imageDataset = tf.data.Dataset.from_tensor_slices(shuffledImageFiles.ToArray())
            .map(filePath => {
                var imageContent = tf.io.read_file(filePath); // Resim dosyasını oku
                var image = tf.image.decode_jpeg(imageContent, channels: 3); // JPEG formatında resmi decode et
                image = tf.image.resize(image, new[] { imageSize, imageSize }); // Resmi yeniden boyutlandır
                image = image / 255.0f; // Normalizasyon işlemi (0-1 arası)
                return image;
            });

        // Etiketlerden TensorFlow Dataset oluştur
        var labelDataset = tf.data.Dataset.from_tensor_slices(labels);

        // Resim ve etiketleri birleştir
        var dataset = tf.data.Dataset.zip(imageDataset, labelDataset)
            .batch(batchSize) // Batch'leri oluştur
            .prefetch(1); // İleriye dönük veriyi önceden yükle

        return dataset;
    }

    private static Sequential BuildModel()
    {
        // Yeni bir Sequential model oluştur
        var model = keras.Sequential();
        model.add(keras.layers.InputLayer(input_shape: (imageSize, imageSize, 3))); // Girdi katmanı

        model.add(keras.layers.Conv2D(16, (3, 3), activation: "relu", padding: "VALID")); // İlk konvolüsyon katmanı
        model.add(keras.layers.MaxPooling2D((2, 2), padding: "VALID")); // İlk max pooling katmanı

        model.add(keras.layers.Conv2D(32, (3, 3), activation: "relu", padding: "VALID")); // İkinci konvolüsyon katmanı
        model.add(keras.layers.MaxPooling2D((2, 2), padding: "VALID")); // İkinci max pooling katmanı

        model.add(keras.layers.Conv2D(128, (3, 3), activation: "relu", padding: "VALID")); // Üçüncü konvolüsyon katmanı
        model.add(keras.layers.MaxPooling2D((2, 2), padding: "VALID")); // Üçüncü max pooling katmanı

        model.add(keras.layers.Flatten()); // Düzleştirme katmanı
        model.add(keras.layers.Dense(128, activation: "relu")); // İlk yoğun katman
        model.add(keras.layers.Dense(64, activation: "relu")); // İkinci yoğun katman
        model.add(keras.layers.Dense(4, activation: "softmax")); // Çıkış katmanı (4 sınıf için softmax)

        model.compile(optimizer: new Adam(), loss: keras.losses.SparseCategoricalCrossentropy(), metrics: new[] { "accuracy" }); // Modeli derle
        model.summary(); // Model özetini yazdır
        return model;
    }

    private static IDictionary<int, float> ComputeClassWeights(IDatasetV2 trainData)
    {
        var classCounts = new Dictionary<int, int>(); // Sınıf sayıları için bir sözlük
        var totalCount = 0; // Toplam örnek sayısı

        // Dataset üzerinden geçiş yap
        foreach (var batch in trainData)
        {
            var labels = batch.Item2.ToArray<int>(); // Etiketleri diziye dönüştür

            // Her bir sınıfın sayısını hesapla
            foreach (var label in labels)
            {
                if (!classCounts.ContainsKey(label))
                    classCounts[label] = 0;
                classCounts[label]++;
                totalCount++;
            }
        }

        var classWeights = new Dictionary<int, float>(); // Sınıf ağırlıkları için bir sözlük
        var numClasses = classCounts.Count; // Toplam sınıf sayısı

        // Sınıf ağırlıklarını hesapla
        foreach (var kv in classCounts)
        {
            var classWeight = totalCount / (float)(numClasses * kv.Value);
            classWeights[kv.Key] = classWeight;
        }

        return classWeights;
    }

    private static void EvaluateModel(Model model, IDatasetV2 testData)
    {
        var evaluation = model.evaluate(testData); // Modeli değerlendirme
        if (evaluation.ContainsKey("loss") && evaluation.ContainsKey("accuracy"))
        {
            float testLoss = evaluation["loss"]; // Test kaybını al
            float testAccuracy = evaluation["accuracy"]; // Test doğruluğunu al

            Console.WriteLine($"Test Loss: {testLoss}"); // Test kaybını yazdır
            Console.WriteLine($"Test Accuracy: {testAccuracy}"); // Test doğruluğunu yazdır
        }
        else
        {
            Console.WriteLine("Test results dictionary does not contain expected keys."); // Sonuç sözlüğü beklenen anahtarları içermiyor
        }

        var predictions = new List<int>(); // Tahminleri saklamak için liste
        var labels = new List<int>(); // Etiketleri saklamak için liste

        foreach (var batch in testData)
        {
            var (x, y) = batch;
            var yPred = model.predict(x); // Tahminleri al

            // Tahminleri NumPy dizisine ve sonra int dizisine dönüştür
            var yPredArray = yPred.numpy();
            var yPredIntArray = np.argmax(yPredArray, axis: 1).astype(np.int32); // Doğru türde int dizisi
            var yPrediction = yPredIntArray.ToArray<int>();

            predictions.AddRange(yPrediction); // Tahminleri listeye ekle

            // Etiketleri int dizisine dönüştür
            var yArray = y.numpy();
            var yIntArray = yArray.astype(np.int32); // Doğru türde int dizisi
            labels.AddRange(yIntArray.ToArray<int>());
        }

        var classificationReport = ClassificationReport(labels.ToArray(), predictions.ToArray()); // Sınıflandırma raporunu al
        Console.WriteLine(classificationReport); // Sınıflandırma raporunu yazdır

        var cm = ConfusionMatrix(labels.ToArray(), predictions.ToArray()); // Karışıklık matrisini al
        PrintConfusionMatrix(cm); // Karışıklık matrisini yazdır
    }

    private static string ClassificationReport(int[] labels, int[] predictions)
    {
        // Etiketler ve tahminlerden sınıflandırma raporu oluşturma yöntemi implementasyonu
        return "Classification report implementation needed";
    }

    private static int[,] ConfusionMatrix(int[] labels, int[] predictions, int numClasses = 4)
    {
        var cm = new int[numClasses, numClasses]; // Karışıklık matrisini tanımla

        // Karışıklık matrisini doldur
        for (int i = 0; i < labels.Length; i++)
        {
            cm[labels[i], predictions[i]]++;
        }

        return cm;
    }

    private static void PrintConfusionMatrix(int[,] cm)
    {
        // Karışıklık matrisini yazdır
        for (int i = 0; i < cm.GetLength(0); i++)
        {
            for (int j = 0; j < cm.GetLength(1); j++)
            {
                Console.Write($"{cm[i, j]} "); // Matris elemanını yazdır
            }
            Console.WriteLine();
        }
    }

    private static void CheckDirectoryContents()
    {
        var datasetDir = new DirectoryInfo(datasetPath); // Dataset dizin bilgisini al

        if (datasetDir.Exists)
        {
            Console.WriteLine($"Contents of {datasetPath}:"); // Dizin içeriğini yazdır
            foreach (var dir in datasetDir.GetDirectories())
            {
                Console.WriteLine($"Directory: {dir.FullName}"); // Alt dizinleri yazdır
            }
        }
        else
        {
            Console.WriteLine($"Directory does not exist: {datasetPath}"); // Dizin mevcut değilse uyarı
        }
    }
}
